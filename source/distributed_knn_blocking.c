#include "matrix.h"
#include "knn.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

/* Provide a blocking variant helper (different symbol names to avoid link conflicts).
 * If you want to use blocking functions, adapt testing.c to call them.
 */

struct KNN_Pair **knn_search_distributed_blocking(matrix_t *local_data, int k,
                                                  int prev_task, int next_task,
                                                  int tasks_num)
{
    int rows = matrix_get_rows(local_data);
    struct KNN_Pair **res = knn_search(local_data, local_data, k+1, matrix_get_chunk_offset(local_data));
    struct KNN_Pair **out = KNN_Pair_create_empty_table(rows, k);
    for (int i = 0; i < rows; ++i) {
        int filled = 0;
        for (int j = 0; j < k+1 && filled < k; ++j) {
            if (res[i][j].index == matrix_get_chunk_offset(local_data) + i) continue;
            out[i][filled].distance = res[i][j].distance;
            out[i][filled].index = res[i][j].index;
            filled++;
        }
        while (filled < k) { out[i][filled].distance = 1e300; out[i][filled].index = -1; filled++; }
    }
    KNN_Pair_destroy_table(res, rows);
    return out;
}

/* blocking send/recv stubs (not used) */
void _send_object_blocking(char *object, size_t length, int rank) {
    MPI_Send(object, (int)length, MPI_CHAR, rank, 0, MPI_COMM_WORLD);
}
void _recv_object_blocking(char **object, size_t *length, int rank) {
    /* not implemented */
}

