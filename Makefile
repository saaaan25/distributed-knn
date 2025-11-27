CC = mpicc
CFLAGS = -O2 -fopenmp -Wall
SRC = source/testing.c source/matrix.c source/knn.c source/distributed_knn.c
TARGET = distributed_knn

all: $(TARGET) knn_secuencial main testing

# === Ejecutable MPI principal (original) ===
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) -lm

# === Nuevo ejecutable main (soporta txt + binario + MPI + OpenMP) ===
main:
	$(CC) $(CFLAGS) source/main.c source/matrix.c source/knn.c source/distributed_knn.c source/distributed_knn_blocking.c -o main -lm

# === Ejecutable de pruebas secuenciales (solo TXT) ===
testing:
	gcc -O2 source/testing.c source/matrix.c source/knn.c -o testing -lm

# === Ejecutable KNN secuencial (TXT) ===
knn_secuencial:
	gcc -O2 source/knn_secuencial.c source/matrix.c source/knn.c -o knn_secuencial -lm

clean:
	rm -f $(TARGET) knn_secuencial main testing
