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
    if (argc != 8) {
    printf("Uso: %s <edad> <estatura> <peso> <glucosa> <fc> <oxigeno> <k>\n", argv[0]);
    return -1;
}

    double edad     = atof(argv[1]);
    double estatura = atof(argv[2]);
    double peso     = atof(argv[3]);
    double glucosa  = atof(argv[4]);
    double fc       = atof(argv[5]);
    double oxigeno  = atof(argv[6]);
    int k           = atoi(argv[7]);

    if (k <= 0) { fprintf(stderr, "k debe ser > 0\n"); return -1; }

    int tasks_num = 1, rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const char *data_fn = "dataset/input.txt"; //dataset con 6 features + 1 label

    /* Each proc loads its chunk */
    matrix_t *local_data = matrix_load_in_chunks(data_fn, tasks_num, rank);
    if (!local_data) {
        if (rank == MPI_MASTER) fprintf(stderr, "ERROR: failed to load dataset %s\n", data_fn);
        MPI_Finalize();
        return -1;
    }

    /* Build query (1 x 6 features) */
    int cols = matrix_get_cols(local_data);
    if (cols < 7) {
        if (rank == MPI_MASTER) fprintf(stderr, "ERROR: dataset debe tener al menos 7 columnas (6 features + label)\n");
        matrix_destroy(local_data);
        MPI_Finalize();
        return -1;
    }
    matrix_t *query = matrix_create(1, cols - 1);
    matrix_set_cell(query, 0, 0, edad);
    matrix_set_cell(query, 0, 1, estatura);
    matrix_set_cell(query, 0, 2, peso);
    matrix_set_cell(query, 0, 3, glucosa);
    matrix_set_cell(query, 0, 4, fc);
    matrix_set_cell(query, 0, 5, oxigeno);

    /* --- Medición de tiempo total --- */
    MPI_Barrier(MPI_COMM_WORLD);
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    /* Each process computes its k nearest neighbors for the single query against its local_data */
    struct KNN_Pair **local_knns = knn_search(local_data, query, k, matrix_get_chunk_offset(local_data));
    if (!local_knns) {
        if (rank == MPI_MASTER) fprintf(stderr, "ERROR: knn_search failed\n");
        matrix_destroy(local_data); matrix_destroy(query);
        MPI_Finalize();
        return -1;
    }

    /* Prepare send buffer: per neighbor: distance, index, x, y, label (5 doubles) */
    int elems_per = 9;
    double *sendbuf = (double*) malloc(sizeof(double) * k * elems_per);
    for (int i = 0; i < k; ++i) {
        double dist = local_knns[0][i].distance;
        int idx = local_knns[0][i].index;
        double features[6] = {NAN,NAN,NAN,NAN,NAN,NAN};
        double lab = NAN;
        int offset = matrix_get_chunk_offset(local_data);
        int local_idx = idx - offset;
        if (local_idx >= 0 && local_idx < matrix_get_rows(local_data)) {
            for (int f = 0; f < 6; f++) {
                features[f] = matrix_get_cell(local_data, local_idx, f);
            }
            lab = matrix_get_cell(local_data, local_idx, 6); // última columna = etiqueta
        }
        sendbuf[elems_per * i + 0] = dist;
        sendbuf[elems_per * i + 1] = (double) idx;
        for (int f = 0; f < 6; f++) {
            sendbuf[elems_per * i + 2 + f] = features[f];
        }
        sendbuf[elems_per * i + 8] = lab;
    }
    
    double *recvbuf = NULL;
    if (rank == MPI_MASTER) {
        recvbuf = (double*) malloc(sizeof(double) * k * elems_per * tasks_num);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t0, NULL);

    MPI_Gather(sendbuf, k*elems_per, MPI_DOUBLE, recvbuf, k*elems_per, MPI_DOUBLE, MPI_MASTER, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t1, NULL);

    if (rank == MPI_MASTER) {
        double elapsed_total = get_elapsed_time(t0, t1);
        printf("\nTiempo total de ejecución del KNN distribuido = %.6f segundos\n", elapsed_total);
        int total = k * tasks_num;
        struct KNN_Pair *all = (struct KNN_Pair*) malloc(sizeof(struct KNN_Pair) * total);
        for (int i = 0; i < total; ++i) {
            all[i].distance = recvbuf[elems_per * i + 0];
            all[i].index = (int) recvbuf[elems_per * i + 1];
        }
        qsort(all, total, sizeof(struct KNN_Pair), KNN_Pair_asc_comp);

        double elapsed = get_elapsed_time(t0, t1);
        printf("Distributed single-query knn usando %d procesos: tiempo gather + sort = %.6f secs\n", tasks_num, elapsed);

        printf("\n=== Top %d vecinos para query (edad=%.1f, estatura=%.1f, peso=%.1f, glucosa=%.1f, fc=%.1f, oxigeno=%.1f) ===\n",
               k, edad, estatura, peso, glucosa, fc, oxigeno);

        for (int i = 0, printed = 0; printed < k && i < total; ++i) {
            int idx = all[i].index;
            double dist = all[i].distance;
            double feats[6] = {NAN,NAN,NAN,NAN,NAN,NAN};
            double rlab = NAN;
            for (int p = 0; p < total; ++p) {
                if ((int) recvbuf[elems_per * p + 1] == idx) {
                    for (int f = 0; f < 6; f++) feats[f] = recvbuf[elems_per * p + 2 + f];
                    rlab = recvbuf[elems_per * p + 8];
                    break;
                }
            }
            printf("%d) idx=%d  edad=%.1f estatura=%.1f peso=%.1f glucosa=%.1f fc=%.1f oxigeno=%.1f  label=%.0f  dist=%.6f\n",
                   printed+1, idx,
                   feats[0], feats[1], feats[2], feats[3], feats[4], feats[5],
                   rlab, dist);
            printed++;
        }

        /* Votación mayoritaria de etiquetas */
        int label_counts[2048] = {0};
        int best_label = -1, best_count = 0;
        for (int i = 0, collected = 0; collected < k && i < total; ++i) {
            int idx = all[i].index;
            double rlab = NAN;
            for (int p = 0; p < total; ++p) {
                if ((int) recvbuf[elems_per * p + 1] == idx) {
                    rlab = recvbuf[elems_per * p + 8];
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

