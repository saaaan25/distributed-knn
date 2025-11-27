#ifndef DISTRIBUTED_KNN_BLOCKING_H
#define DISTRIBUTED_KNN_BLOCKING_H

#include "matrix.h"
#include "knn.h"

struct KNN_Pair **knn_search_distributed(
    matrix_t *local_data,
    int k,
    int prev_task,
    int next_task,
    int tasks_num
);

void _send_object(char *object, size_t length, int rank);
void _recv_object(char **object, size_t *length, int rank);
void _update_knns(struct KNN_Pair **original, struct KNN_Pair **new, int points, int k);

#endif

