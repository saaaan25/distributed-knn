#include "knn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/* helpers for KNN_Pair table */
struct KNN_Pair **KNN_Pair_create_empty_table(int points, int k) {
    struct KNN_Pair **table = (struct KNN_Pair**) malloc(sizeof(struct KNN_Pair*) * points);
    if (!table) return NULL;
    for (int i = 0; i < points; ++i) {
        table[i] = (struct KNN_Pair*) malloc(sizeof(struct KNN_Pair) * k);
        if (!table[i]) {
            for (int j = 0; j < i; ++j) free(table[j]);
            free(table);
            return NULL;
        }
        for (int j = 0; j < k; ++j) { table[i][j].distance = 1e300; table[i][j].index = -1; }
    }
    return table;
}

void KNN_Pair_destroy_table(struct KNN_Pair **table, int rows) {
    if (!table) return;
    for (int i = 0; i < rows; ++i) free(table[i]);
    free(table);
}

int KNN_Pair_asc_comp(const void *a, const void *b) {
    double da = ((struct KNN_Pair*)a)->distance;
    double db = ((struct KNN_Pair*)b)->distance;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}
int KNN_Pair_asc_comp_by_index(const void *a, const void *b) {
    return ((struct KNN_Pair*)a)->index - ((struct KNN_Pair*)b)->index;
}

/* knn_search: use only first two columns for coords (x,y). That avoids using label column. */
struct KNN_Pair **knn_search(matrix_t *data, matrix_t *points, int k, int i_offset) {
    if (!data || !points || k < 1) return NULL;

    int data_rows = matrix_get_rows(data);
    int P = matrix_get_rows(points);
    int dims = matrix_get_cols(points);
    if (dims < 2) dims = 2;

    struct KNN_Pair **results = KNN_Pair_create_empty_table(P, k);
    if (!results) return NULL;

    // Paralelismo
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < P; ++p) {

        struct KNN_Pair *local_knn = results[p];

        for (int d = 0; d < data_rows; ++d) {

            // Calcula distancia euclidiana
            double dist = 0.0;
            for (int c = 0; c < 2; ++c) {
                double diff = matrix_get_cell(points, p, c)
                             - matrix_get_cell(data, d, c);
                dist += diff * diff;
            }
            dist = sqrt(dist);

            if (local_knn[k-1].index == -1 || dist < local_knn[k-1].distance) {
                local_knn[k-1].distance = dist;
                local_knn[k-1].index = i_offset + d;

                qsort(local_knn, k, sizeof(struct KNN_Pair), KNN_Pair_asc_comp);
            }
        }
    }

    return results;
}

/* knn_labeling: for each knn pair pick label from labels matrix if available (labels expected in column 0) */
matrix_t *knn_labeling(struct KNN_Pair **knns, int points, int k,
                       matrix_t *previous, int *cur_indexes,
                       matrix_t *labels, int i_offset) {
    if (!knns || !labels) return NULL;
    matrix_t *labeled = previous;
    if (!labeled) labeled = matrix_create(points, k);
    for (int p = 0; p < points; ++p) {
        for (int j = 0; j < k; ++j) {
            int global_idx = knns[p][j].index;
            int local_idx = global_idx - i_offset;
            double lab = 0.0;
            if (local_idx >= 0 && local_idx < matrix_get_rows(labels)) {
                lab = matrix_get_cell(labels, local_idx, 0);
            } else {
                lab = NAN;
            }
            matrix_set_cell(labeled, p, j, lab);
        }
    }
    return labeled;
}

/* classify: majority vote among integer labels (ignoring NaN) */
matrix_t *knn_classify(matrix_t *labeled_knns) {
    int rows = matrix_get_rows(labeled_knns);
    int cols = matrix_get_cols(labeled_knns);
    matrix_t *out = matrix_create(rows, 1);
    for (int r = 0; r < rows; ++r) {
        int maxlabel = 0;
        for (int c = 0; c < cols; ++c) {
            double v = matrix_get_cell(labeled_knns, r, c);
            if (!isnan(v)) {
                int iv = (int) v;
                if (iv > maxlabel) maxlabel = iv;
            }
        }
        int buckets = maxlabel + 2;
        if (buckets < 8) buckets = 8;
        int *counts = (int*) calloc(buckets, sizeof(int));
        for (int c = 0; c < cols; ++c) {
            double v = matrix_get_cell(labeled_knns, r, c);
            if (!isnan(v)) {
                int iv = (int) v;
                if (iv >= 0 && iv < buckets) counts[iv]++;
            }
        }
        int best = 0, bestc = counts[0];
        for (int i = 1; i < buckets; ++i) {
            if (counts[i] > bestc) { best = i; bestc = counts[i]; }
        }
        matrix_set_cell(out, r, 0, (double) best);
        free(counts);
    }
    return out;
}

