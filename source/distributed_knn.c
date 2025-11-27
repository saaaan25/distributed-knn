#include "matrix.h"
#include "knn.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>

/* simplified distributed knn: here we implement a helper that for each process
 * computes knn of its local_data vs local_data (removing self), producing table rows x k.
 * Real distribution (ring passing) is more complex; this keeps interface compatible.
 */

struct KNN_Pair **knn_search_distributed(matrix_t *local_data, int k,
                                         int prev_task, int next_task,
                                         int tasks_num)
{
    int rows = matrix_get_rows(local_data);
    /* compute knn on local_data vs local_data (k+1 to skip self) */
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

/* minimal stubs for async helpers used previously (kept for linking) */
MPI_Request *_async_send_object(char *object, size_t length, int rank, int *handlerc) {
    MPI_Request *req = (MPI_Request*) malloc(sizeof(MPI_Request));
    *handlerc = 1;
    MPI_Isend(object, (int)length, MPI_CHAR, rank, 0, MPI_COMM_WORLD, req);
    return req;
}
MPI_Request *_async_recv_object(char **object, size_t *length, int rank, int *handlerc) {
    MPI_Request *req = (MPI_Request*) malloc(sizeof(MPI_Request));
    *handlerc = 1;
    /* Not implementing full async protocol here */
    return req;
}
void _wait_async_com(MPI_Request *handlers, int handlerc) {
    if (!handlers) return;
    MPI_Waitall(handlerc, handlers, MPI_STATUSES_IGNORE);
}
void _update_knns(struct KNN_Pair **original, struct KNN_Pair **new, int points, int k) {
    for (int i = 0; i < points; ++i) {
        int total = k * 2;
        struct KNN_Pair *both = (struct KNN_Pair*) malloc(sizeof(struct KNN_Pair) * total);
        for (int a = 0; a < k; ++a) both[a] = original[i][a];
        for (int a = 0; a < k; ++a) both[k+a] = new[i][a];
        qsort(both, total, sizeof(struct KNN_Pair), KNN_Pair_asc_comp);
        for (int a = 0; a < k; ++a) original[i][a] = both[a];
        free(both);
    }
}

