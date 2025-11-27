#ifndef KNN_H
#define KNN_H

#include "matrix.h"

struct KNN_Pair {
    double distance;
    int index;
};

struct KNN_Pair **KNN_Pair_create_empty_table(int points, int k);
void KNN_Pair_destroy_table(struct KNN_Pair **table, int rows);

int KNN_Pair_asc_comp(const void *a, const void *b);
int KNN_Pair_asc_comp_by_index(const void *a, const void *b);

struct KNN_Pair **knn_search(matrix_t *data, matrix_t *points, int k, int i_offset);

matrix_t *knn_labeling(struct KNN_Pair **knns, int points, int k,
                       matrix_t *previous, int *cur_indexes,
                       matrix_t *labels, int i_offset);

matrix_t *knn_classify(matrix_t *labeled_knns);

#endif

