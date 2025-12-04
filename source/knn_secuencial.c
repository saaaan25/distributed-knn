#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_POINTS 10000000
#define FEATURES 6   // número de columnas de características

typedef struct {
    double features[FEATURES];  // edad, estatura, peso, glucosa, FC, oxígeno
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

double euclidean_distance(const double *a, const double *b) {
    double sum = 0.0;
    for (int i = 0; i < FEATURES; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int main(int argc, char *argv[]) {

    if (argc != FEATURES + 2) {
        printf("Uso: ./knn_secuencial ");
        for (int i = 0; i < FEATURES; i++) printf("v%d ", i+1);
        printf(" k\n");
        printf("Ejemplo: ./knn_secuencial 30 170 75 110 80 97 5\n");
        return 1;
    }

    // Leer punto de consulta
    double query[FEATURES];
    for (int i = 1; i <= FEATURES; i++)
        query[i-1] = atof(argv[i]);

    int k = atoi(argv[FEATURES + 1]);

    FILE *fp = fopen("../dataset/pacientes_dataset.txt", "r");
    if (!fp) {
        printf("ERROR: No se pudo abrir ../dataset/pacientes_dataset.txt\n");
        return 1;
    }

    Point dataset[MAX_POINTS];
    int count = 0;

    // Leer dataset genérico de 7 columnas
    while (fscanf(fp,
                  "%lf %lf %lf %lf %lf %lf %d",
                  &dataset[count].features[0],
                  &dataset[count].features[1],
                  &dataset[count].features[2],
                  &dataset[count].features[3],
                  &dataset[count].features[4],
                  &dataset[count].features[5],
                  &dataset[count].label) == 7) {
        count++;
        if (count >= MAX_POINTS) break;
    }
    fclose(fp);

    if (k > count) {
        printf("ERROR: k es mayor al número de puntos (%d).\n", count);
        return 1;
    }

    Neighbor *neighbors = malloc(sizeof(Neighbor) * count);

    // Calcular distancias
    for (int i = 0; i < count; i++) {
        neighbors[i].index = i;
        neighbors[i].distance = euclidean_distance(query, dataset[i].features);
    }

    // Ordenar por distancia ascendente
    qsort(neighbors, count, sizeof(Neighbor), compare_neighbors);

    // Mostrar vecinos más cercanos
    printf("\nPunto consultado:\n");
    for (int i = 0; i < FEATURES; i++)
        printf("  v%d = %.3f\n", i+1, query[i]);

    printf("\nk = %d\n\n", k);

    printf("Vecinos más cercanos:\n");
    for (int i = 0; i < k; i++) {
        int idx = neighbors[i].index;
        printf("%d) dist=%.5f label=%d  -> [",
               i+1, neighbors[i].distance, dataset[idx].label);

        for (int j = 0; j < FEATURES; j++)
            printf("%.2f%s", dataset[idx].features[j], (j<FEATURES-1 ? ", " : ""));

        printf("]\n");
    }

    // Clasificación por mayoría
    int class_count[100] = {0};
    for (int i = 0; i < k; i++) {
        int lbl = dataset[neighbors[i].index].label;
        class_count[lbl]++;
    }

    int best_class = 0, best_count = class_count[0];
    for (int i = 1; i < 100; i++) {
        if (class_count[i] > best_count) {
            best_count = class_count[i];
            best_class = i;
        }
    }

    printf("\nClase predicha: %d\n\n", best_class);

    free(neighbors);
    return 0;
}
