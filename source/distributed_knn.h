#ifndef DISTRIBUTED_KNN_H
#define DISTRIBUTED_KNN_H

#include "matrix.h"
#include "knn.h"

struct KNN_Pair **knn_search_distributed(
    matrix_t *local_data,
    int k,
    int prev_task,
    int next_task,
    int tasks_num
);

matrix_t *knn_labeling_distributed(
    struct KNN_Pair **knns,
    int points,
    int k,
    matrix_t *labels,
    int prev_task,
    int next_task,
    int tasks_num
);


/* Asynchronous helpers (not actually used but declared) */
MPI_Request *_async_send_object(char *object, size_t length, int rank, int *handlerc);
MPI_Request *_async_recv_object(char **object, size_t *length, int rank, int *handlerc);
void _wait_async_com(MPI_Request *handlers, int handlerc);
void _update_knns(struct KNN_Pair **original, struct KNN_Pair **new, int points, int k);

#endif

