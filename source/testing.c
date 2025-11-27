#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#include "matrix.h"
#include "knn.h"

#define MPI_MASTER 0

double get_elapsed_time(struct timeval start, struct timeval stop) {
    double elapsed_time = (stop.tv_sec - start.tv_sec) * 1.0;
    elapsed_time += (stop.tv_usec - start.tv_usec) / 1000000.0;
    return elapsed_time;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <query_x> <query_y> <k>\n", argv[0]);
        return -1;
    }

    double qx = atof(argv[1]);
    double qy = atof(argv[2]);
    int k = atoi(argv[3]);
    if (k <= 0) { fprintf(stderr, "k debe ser > 0\n"); return -1; }

    int tasks_num = 1, rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const char *data_fn = "dataset/input.txt";

    /* Each proc loads its chunk */
    matrix_t *local_data = matrix_load_in_chunks(data_fn, tasks_num, rank);
    if (!local_data) {
        if (rank == MPI_MASTER) fprintf(stderr, "ERROR: failed to load dataset %s\n", data_fn);
        MPI_Finalize();
        return -1;
    }

    /* Build query (1 x cols) */
    int cols = matrix_get_cols(local_data);
    if (cols < 2) {
        if (rank == MPI_MASTER) fprintf(stderr, "ERROR: dataset must have at least 2 columns (x y)\n");
        matrix_destroy(local_data);
        MPI_Finalize();
        return -1;
    }
    matrix_t *query = matrix_create(1, cols);
    matrix_set_cell(query, 0, 0, qx);
    matrix_set_cell(query, 0, 1, qy);
    for (int c = 2; c < cols; ++c) matrix_set_cell(query, 0, c, 0.0);

    /* Each process computes its k nearest neighbors for the single query against its local_data */
    struct KNN_Pair **local_knns = knn_search(local_data, query, k, matrix_get_chunk_offset(local_data));
    if (!local_knns) {
        if (rank == MPI_MASTER) fprintf(stderr, "ERROR: knn_search failed\n");
        matrix_destroy(local_data); matrix_destroy(query);
        MPI_Finalize();
        return -1;
    }

    /* Prepare send buffer: per neighbor: distance, index, x, y, label (5 doubles) */
    int elems_per = 5;
    double *sendbuf = (double*) malloc(sizeof(double) * k * elems_per);
    for (int i = 0; i < k; ++i) {
        double dist = local_knns[0][i].distance;
        int idx = local_knns[0][i].index;
        double x = NAN, y = NAN, lab = NAN;
        int offset = matrix_get_chunk_offset(local_data);
        int local_idx = idx - offset;
        if (local_idx >= 0 && local_idx < matrix_get_rows(local_data)) {
            x = matrix_get_cell(local_data, local_idx, 0);
            y = matrix_get_cols(local_data) > 1 ? matrix_get_cell(local_data, local_idx, 1) : NAN;
            if (matrix_get_cols(local_data) > 2) lab = matrix_get_cell(local_data, local_idx, 2);
        }
        sendbuf[elems_per * i + 0] = dist;
        sendbuf[elems_per * i + 1] = (double) idx;
        sendbuf[elems_per * i + 2] = x;
        sendbuf[elems_per * i + 3] = y;
        sendbuf[elems_per * i + 4] = lab;
    }

    double *recvbuf = NULL;
    if (rank == MPI_MASTER) {
        recvbuf = (double*) malloc(sizeof(double) * k * elems_per * tasks_num);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    MPI_Gather(sendbuf, k*elems_per, MPI_DOUBLE, recvbuf, k*elems_per, MPI_DOUBLE, MPI_MASTER, MPI_COMM_WORLD);

    gettimeofday(&t1, NULL);

    if (rank == MPI_MASTER) {
        int total = k * tasks_num;
        struct KNN_Pair *all = (struct KNN_Pair*) malloc(sizeof(struct KNN_Pair) * total);
        for (int i = 0; i < total; ++i) {
            all[i].distance = recvbuf[elems_per * i + 0];
            all[i].index = (int) recvbuf[elems_per * i + 1];
        }
        qsort(all, total, sizeof(struct KNN_Pair), KNN_Pair_asc_comp);

        double elapsed = get_elapsed_time(t0, t1);
        printf("Distributed single-query knn using %d processes: gather + local sort time = %.6f secs\n", tasks_num, elapsed);

        printf("\n=== Top %d neighbors for (%.4f, %.4f) ===\n", k, qx, qy);
        for (int i = 0, printed = 0; printed < k && i < total; ++i) {
            int idx = all[i].index;
            double dist = all[i].distance;
            double rx = NAN, ry = NAN, rlab = NAN;
            for (int p = 0; p < total; ++p) {
                if ((int) recvbuf[elems_per * p + 1] == idx) {
                    rx = recvbuf[elems_per * p + 2];
                    ry = recvbuf[elems_per * p + 3];
                    rlab = recvbuf[elems_per * p + 4];
                    break;
                }
            }
            printf("%d) idx=%d  (%.6f, %.6f)  label=%.0f  dist=%.6f\n",
                   printed+1, idx, rx, ry, rlab, dist);
            printed++;
        }

        /* majority vote from top-k labels (ignore NaN), if none -> -1 */
        int label_counts[2048] = {0};
        int best_label = -1, best_count = 0;
        for (int i = 0, collected = 0; collected < k && i < total; ++i) {
            int idx = all[i].index;
            double rlab = NAN;
            for (int p = 0; p < total; ++p) {
                if ((int) recvbuf[elems_per * p + 1] == idx) {
                    rlab = recvbuf[elems_per * p + 4];
                    break;
                }
            }
            if (!isnan(rlab)) {
                int li = (int) rlab;
                if (li >= 0 && li < 2048) {
                    label_counts[li]++;
                    if (label_counts[li] > best_count) { best_count = label_counts[li]; best_label = li; }
                }
                collected++;
            }
        }
        printf("\nPredicted class: %d (votes=%d)\n", best_label, best_count);
        free(all);
    }

    /* cleanup */
    if (recvbuf) free(recvbuf);
    free(sendbuf);
    KNN_Pair_destroy_table(local_knns, 1);
    matrix_destroy(local_data);
    matrix_destroy(query);

    MPI_Finalize();
    return 0;
}

