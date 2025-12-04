#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_POINTS 10000000
#define DIM 6   // 6 características

typedef struct {
    double features[DIM];  // edad, estatura, peso, glucosa, fc, oxígeno
    int label;
} Point;

typedef struct {
    int index;
    double distance;
} Neighbor;

// Comparador para ordenar vecinos
int compare_neighbors(const void *a, const void *b) {
    Neighbor *na = (Neighbor *)a;
    Neighbor *nb = (Neighbor *)b;
    return (na->distance > nb->distance) - (na->distance < nb->distance);
}

// Distancia euclidiana en 6 dimensiones
double distance(double q[], double p[]) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++) {
        double diff = q[i] - p[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int main(int argc, char *argv[]) {

    if (argc != 8) {
        printf("Uso: ./knn_secuencial edad estatura peso glucosa frecuencia oxigeno k\n");
        return 1;
    }

    // Leer punto query desde argumentos
    double query[DIM];
    for (int i = 0; i < DIM; i++)
        query[i] = atof(argv[i + 1]);

    int k = atoi(argv[7]);

    if (k <= 0) {
        printf("ERROR: k debe ser > 0\n");
        return 1;
    }

    FILE *fp = fopen("dataset.txt", "r");
    if (!fp) {
        printf("ERROR: No se pudo abrir dataset.txt\n");
        return 1;
    }

    Point dataset[MAX_POINTS];
    int count = 0;

    // Leer dataset (6 features + 1 label)
    while (fscanf(fp, "%lf %lf %lf %lf %lf %lf %d",
                  &dataset[count].features[0],
                  &dataset[count].features[1],
                  &dataset[count].features[2],
                  &dataset[count].features[3],
                  &dataset[count].features[4],
                  &dataset[count].features[5],
                  &dataset[count].label) == 7)
    {
        count++;
        if (count >= MAX_POINTS) break;
    }
    fclose(fp);

    if (k > count) {
        printf("ERROR: k es mayor al tamaño del dataset.\n");
        return 1;
    }

    Neighbor *neighbors = malloc(sizeof(Neighbor) * count);

    // Calcular distancias
    for (int i = 0; i < count; i++) {
        neighbors[i].index = i;
        neighbors[i].distance = distance(query, dataset[i].features);
    }

    // Ordenar por distancia
    qsort(neighbors, count, sizeof(Neighbor), compare_neighbors);

    // Mostrar vecinos más cercanos
    printf("\nPunto ingresado:\n");
    for (int i = 0; i < DIM; i++)
        printf("%.3f ", query[i]);
    printf("\nk = %d\n\n", k);

    printf("Vecinos más cercanos:\n");
    for (int i = 0; i < k; i++) {
        int idx = neighbors[i].index;
        printf("%d) dist=%.5f  label=%d | %.1f %.1f %.1f %.1f %.1f %.1f\n",
               i + 1,
               neighbors[i].distance,
               dataset[idx].label,
               dataset[idx].features[0],
               dataset[idx].features[1],
               dataset[idx].features[2],
               dataset[idx].features[3],
               dataset[idx].features[4],
               dataset[idx].features[5]);
    }

    // Clasificación (mayoría)
    int count_label_0 = 0, count_label_1 = 0;

    for (int i = 0; i < k; i++) {
        if (dataset[neighbors[i].index].label == 0)
            count_label_0++;
        else
            count_label_1++;
    }

    int predicted = (count_label_1 > count_label_0 ? 1 : 0);

    printf("\nClase predicha = %d (0=no riesgo, 1=riesgo)\n\n", predicted);

    free(neighbors);
    return 0;
}
