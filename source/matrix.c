#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* create/destroy */
matrix_t *matrix_create(int32_t rows, int32_t cols) {
    matrix_t *m = (matrix_t*) malloc(sizeof(matrix_t));
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    m->chunk_offset = 0;
    m->data = (double**) malloc(sizeof(double*) * rows);
    if (!m->data) { free(m); return NULL; }
    for (int32_t i = 0; i < rows; ++i) {
        m->data[i] = (double*) malloc(sizeof(double) * cols);
        if (!m->data[i]) {
            for (int32_t j = 0; j < i; ++j) free(m->data[j]);
            free(m->data);
            free(m);
            return NULL;
        }
        for (int32_t j = 0; j < cols; ++j) m->data[i][j] = 0.0;
    }
    return m;
}

void matrix_destroy(matrix_t *m) {
    if (!m) return;
    for (int32_t i = 0; i < m->rows; ++i) free(m->data[i]);
    free(m->data);
    free(m);
}

/* accessors */
int32_t matrix_get_rows(matrix_t *m) { return m->rows; }
int32_t matrix_get_cols(matrix_t *m) { return m->cols; }
int32_t matrix_get_chunk_offset(matrix_t *m) { return m->chunk_offset; }
double matrix_get_cell(matrix_t *m, int32_t r, int32_t c) { return m->data[r][c]; }
void matrix_set_cell(matrix_t *m, int32_t r, int32_t c, double value) { m->data[r][c] = value; }

/* helper to count non-empty lines and tokens */
static int count_txt(const char *filename, int *out_rows, int *out_cols) {
    FILE *f = fopen(filename, "r");
    if (!f) return -1;
    char line[8192];
    int rows = 0;
    int cols = -1;
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;
        int c = 0;
        char *tok = strtok(line, " \t\r\n");
        while (tok) { c++; tok = strtok(NULL, " \t\r\n"); }
        if (c == 0) continue;
        if (cols == -1) cols = c;
        rows++;
    }
    fclose(f);
    if (cols == -1) cols = 0;
    *out_rows = rows;
    *out_cols = cols;
    return 0;
}

/* load .karas binary or .txt text */
matrix_t *matrix_load_in_chunks(const char *filename, int32_t chunks_num, int32_t req_chunk) {
    int len = (int) strlen(filename);
    int is_txt = 0;
    if (len > 4 && strcmp(filename + len - 4, ".txt") == 0) is_txt = 1;

    if (!is_txt) {
        /* binary .karas format */
        FILE *f = fopen(filename, "rb");
        if (!f) { fprintf(stderr, "ERROR: matrix_load_in_chunks: cannot open %s\n", filename); return NULL; }
        int32_t total_rows = 0, cols = 0;
        if (fread(&total_rows, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
        if (fread(&cols, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
        int32_t rows = total_rows / chunks_num;
        int remaining = total_rows % chunks_num;
        long offset;
        if (req_chunk < remaining) { rows++; offset = req_chunk * rows; }
        else { offset = ((rows + 1) * remaining) + (rows * (req_chunk - remaining)); }
        if (fseek(f, sizeof(double) * offset * cols, SEEK_CUR) != 0) { fclose(f); return NULL; }
        matrix_t *mat = matrix_create(rows, cols);
        mat->chunk_offset = (int32_t) offset;
        for (int32_t i = 0; i < rows; ++i) {
            if (fread(mat->data[i], sizeof(double), cols, f) != (size_t)cols) {
                fprintf(stderr, "ERROR reading binary data\n");
                fclose(f);
                matrix_destroy(mat);
                return NULL;
            }
        }
        fclose(f);
        return mat;
    } else {
        int total_rows = 0, cols = 0;
        if (count_txt(filename, &total_rows, &cols) != 0) { fprintf(stderr, "ERROR: cannot parse %s\n", filename); return NULL; }
        if (chunks_num <= 0) chunks_num = 1;
        int32_t base_rows = total_rows / chunks_num;
        int remaining = total_rows % chunks_num;
        int32_t rows;
        long offset;
        if (req_chunk < remaining) { rows = base_rows + 1; offset = req_chunk * rows; }
        else { rows = base_rows; offset = ((base_rows + 1) * remaining) + (base_rows * (req_chunk - remaining)); }
        matrix_t *mat = matrix_create(rows, cols);
        mat->chunk_offset = (int32_t) offset;
        FILE *f = fopen(filename, "r");
        if (!f) { matrix_destroy(mat); return NULL; }
        char line[8192];
        int cur = 0;
        int filled = 0;
        while (fgets(line, sizeof(line), f)) {
            char *p = line;
            while (*p && isspace((unsigned char)*p)) p++;
            if (*p == '\0' || *p == '#') continue;
            if (cur < offset) { cur++; continue; }
            if (filled >= rows) break;
            int col = 0;
            char *tok = strtok(line, " \t\r\n");
            while (tok && col < cols) {
                mat->data[filled][col] = atof(tok);
                col++;
                tok = strtok(NULL, " \t\r\n");
            }
            for (; col < cols; ++col) mat->data[filled][col] = 0.0;
            filled++;
            cur++;
        }
        fclose(f);
        if (filled != rows) {
            if (filled == 0) { matrix_destroy(mat); return NULL; }
            matrix_t *newm = matrix_create(filled, cols);
            newm->chunk_offset = mat->chunk_offset;
            for (int i = 0; i < filled; ++i)
                for (int j = 0; j < cols; ++j) newm->data[i][j] = mat->data[i][j];
            matrix_destroy(mat);
            mat = newm;
        }
        return mat;
    }
}

/* serialize/deserialize */
char *matrix_serialize(matrix_t *matrix, size_t *bytec) {
    int32_t rows = matrix_get_rows(matrix);
    int32_t cols = matrix_get_cols(matrix);
    int32_t offset = matrix_get_chunk_offset(matrix);
    *bytec = sizeof(int32_t)*3 + sizeof(double)*rows*cols;
    char *buf = (char*) malloc(*bytec);
    char *p = buf;
    memcpy(p, &rows, sizeof(int32_t)); p += sizeof(int32_t);
    memcpy(p, &cols, sizeof(int32_t)); p += sizeof(int32_t);
    memcpy(p, &offset, sizeof(int32_t)); p += sizeof(int32_t);
    for (int i = 0; i < rows; ++i) {
        memcpy(p, matrix->data[i], sizeof(double)*cols);
        p += sizeof(double)*cols;
    }
    return buf;
}

matrix_t *matrix_deserialize(char *bytes, size_t bytec) {
    char *p = bytes;
    int32_t rows, cols, offset;
    memcpy(&rows, p, sizeof(int32_t)); p += sizeof(int32_t);
    memcpy(&cols, p, sizeof(int32_t)); p += sizeof(int32_t);
    memcpy(&offset, p, sizeof(int32_t)); p += sizeof(int32_t);
    matrix_t *m = matrix_create(rows, cols);
    m->chunk_offset = offset;
    for (int i = 0; i < rows; ++i) {
        memcpy(m->data[i], p, sizeof(double)*cols);
        p += sizeof(double)*cols;
    }
    return m;
}

