#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#include "matrix.h"
#include "knn.h"

#define MPI_MASTER 0

double get_elapsed_time(struct timeval start, struct timeval stop) {
    double elapsed_time = (stop.tv_sec - start.tv_sec) * 1.0;
    elapsed_time += (stop.tv_usec - start.tv_usec) / 1000000.0;
    return elapsed_time;
}

int main(int argc, char *argv[]) {

    if (argc < 4) {
        printf("Uso: %s <data_file> <labels_file> <k>\n", argv[0]);
        return -1;
    }

    char *data_fn   = argv[1];
    char *labels_fn = argv[2];
    int k = atoi(argv[3]);

    if (k <= 0) {
        fprintf(stderr, "ERROR: k debe ser > 0\n");
        return -1;
    }

    int tasks_num = 1;
    int rank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int next_task = (rank + 1) % tasks_num;
    int prev_task = (rank == 0 ? tasks_num - 1 : rank - 1);
    
    if (rank == MPI_MASTER) {
        printf("OMP_NUM_THREADS = %s\n",
            getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "not set");
        printf("Hilos disponibles: %d\n", omp_get_max_threads());
        printf("Hilos activos por proceso: %d\n", omp_get_num_threads());
        printf("================================\n\n");
    }

    // LOAD DATA
    matrix_t *initial_data = matrix_load_in_chunks(data_fn, tasks_num, rank);
    if (!initial_data) {
        fprintf(stderr, "ERROR: rank %d no pudo cargar %s\n", rank, data_fn);
        MPI_Finalize();
        return -1;
    }

    // LOAD LABELS
    matrix_t *labels = matrix_load_in_chunks(labels_fn, tasks_num, rank);
    if (!labels) {
        fprintf(stderr, "ERROR: rank %d no pudo cargar %s\n", rank, labels_fn);
        matrix_destroy(initial_data);
        MPI_Finalize();
        return -1;
    }

    // KNN SEARCH
    struct timeval t0, t1;
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t0, NULL);

    struct KNN_Pair **results =
        knn_search_distributed(initial_data, k, prev_task, next_task, tasks_num);

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t1, NULL);

    if (rank == MPI_MASTER) {
        printf("KNN search (%d procesos) tomó %.6f segundos\n",
               tasks_num, get_elapsed_time(t0, t1));
    }

    // LABELING
    matrix_t *labeled =
        knn_labeling_distributed(results,
                                 matrix_get_rows(initial_data),
                                 k, labels,
                                 prev_task, next_task, tasks_num);

    KNN_Pair_destroy_table(results, matrix_get_rows(initial_data));

    // CLASSIFY
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t0, NULL);

    int local_points = matrix_get_rows(labeled);
    matrix_t *classified = matrix_create(local_points, 1);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_points; i++) {
        int label_counts[256] = {0};  

        for (int j = 0; j < k; j++) {
            double v = matrix_get_cell(labeled, i, j);
            if (!isnan(v)) {
                int lab = (int)v;
                if (lab >= 0 && lab < 256)
                    label_counts[lab]++;
            }
        }

        int best = 0, bestc = 0;
        for (int c = 0; c < 256; c++) {
            if (label_counts[c] > bestc) {
                best = c;
                bestc = label_counts[c];
            }
        }
        matrix_set_cell(classified, i, 0, (double)best);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t1, NULL);

    if (rank == MPI_MASTER) {
        printf("Clasificación (OpenMP) tomó %.6f segundos\n",
               get_elapsed_time(t0, t1));
    }

    // VERIFY LOCAL ACCURACY
    int correct = 0;
    for (int i = 0; i < local_points; i++) {
        if (matrix_get_cell(classified, i, 0) ==
            matrix_get_cell(labels, i, 0))
            correct++;
    }

    int total_correct = 0;
    int total_points = 0;

    MPI_Reduce(&correct, &total_correct, 1, MPI_INT, MPI_SUM, MPI_MASTER, MPI_COMM_WORLD);
    MPI_Reduce(&local_points, &total_points, 1, MPI_INT, MPI_SUM, MPI_MASTER, MPI_COMM_WORLD);

    if (rank == MPI_MASTER) {
        double acc = (double)total_correct / total_points * 100.0;
        printf("Accuracy final = %.2f%%\n", acc);
    }

    matrix_destroy(initial_data);
    matrix_destroy(labels);
    matrix_destroy(labeled);
    matrix_destroy(classified);

    MPI_Finalize();
    return 0;
}
