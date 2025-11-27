#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_POINTS 10000

typedef struct {
    double x, y;
    int label;
} Point;

typedef struct {
    int index;
    double distance;
} Neighbor;

int compare_neighbors(const void *a, const void *b) {
    Neighbor *na = (Neighbor *)a;
    Neighbor *nb = (Neighbor *)b;
    return (na->distance > nb->distance) - (na->distance < nb->distance);
}

double distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: ./knn_secuencial x y k\n");
        return 1;
    }

    double qx = atof(argv[1]);
    double qy = atof(argv[2]);
    int k = atoi(argv[3]);

    FILE *fp = fopen("../dataset/input.txt", "r");
    if (!fp) {
        printf("ERROR: No se pudo abrir ../dataset/input.txt\n");
        return 1;
    }

    Point dataset[MAX_POINTS];
    int count = 0;

    // Leer dataset
    while (fscanf(fp, "%lf %lf %d", &dataset[count].x, &dataset[count].y, &dataset[count].label) == 3) {
        count++;
        if (count >= MAX_POINTS) break;
    }
    fclose(fp);

    if (k > count) {
        printf("ERROR: k es mayor al número de puntos en el dataset.\n");
        return 1;
    }

    Neighbor neighbors[MAX_POINTS];

    // Calcular distancias
    for (int i = 0; i < count; i++) {
        neighbors[i].index = i;
        neighbors[i].distance = distance(qx, qy, dataset[i].x, dataset[i].y);
    }

    // Ordenar por distancia
    qsort(neighbors, count, sizeof(Neighbor), compare_neighbors);

    // Imprimir información
    printf("\nPunto: (%.3f, %.3f)\n", qx, qy);
    printf("k = %d\n\n", k);

    printf("Vecinos más cercanos:\n");
    for (int i = 0; i < k; i++) {
        int idx = neighbors[i].index;
        printf("%d) (%.3f, %.3f) label=%d dist=%.5f\n",
               i + 1, dataset[idx].x, dataset[idx].y,
               dataset[idx].label, neighbors[i].distance);
    }

    // Clasificación (mayoría)
    int class_count[100] = {0};
    for (int i = 0; i < k; i++) {
        int lbl = dataset[neighbors[i].index].label;
        class_count[lbl]++;
    }

    // Determinar clase mayoritaria
    int best_class = 0;
    int best_count = class_count[0];
    for (int i = 1; i < 100; i++) {
        if (class_count[i] > best_count) {
            best_class = i;
            best_count = class_count[i];
        }
    }

    printf("\nClase predicha: %d\n\n", best_class);

    return 0;
}

