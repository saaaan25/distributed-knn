#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdlib.h>

typedef struct matrix_t {
    int32_t rows;
    int32_t cols;
    double **data;
    int32_t chunk_offset;
} matrix_t;

matrix_t *matrix_create(int32_t rows, int32_t cols);
void matrix_destroy(matrix_t *m);

int32_t matrix_get_rows(matrix_t *m);
int32_t matrix_get_cols(matrix_t *m);
int32_t matrix_get_chunk_offset(matrix_t *m);

double matrix_get_cell(matrix_t *m, int32_t r, int32_t c);
void matrix_set_cell(matrix_t *m, int32_t r, int32_t c, double v);

matrix_t *matrix_load_in_chunks(const char *filename, int32_t chunks_num, int32_t req_chunk);

char *matrix_serialize(matrix_t *matrix, size_t *bytec);
matrix_t *matrix_deserialize(char *bytes, size_t bytec);

#endif

