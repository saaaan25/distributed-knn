CC = mpicc
CFLAGS = -O2 -fopenmp -Wall
LDFLAGS = -lm

COMMON_SRC = source/matrix.c source/knn.c source/distributed_knn.c source/distributed_knn_blocking.c

all: knn_secuencial testing main

# secuencial
knn_secuencial:
	gcc -O2 source/knn_secuencial.c source/matrix.c source/knn.c -o knn_secuencial -lm

# testing.c
testing:
	$(CC) $(CFLAGS) source/testing.c $(COMMON_SRC) -o testing $(LDFLAGS)

# main.c

main:
	$(CC) $(CFLAGS) source/main.c $(COMMON_SRC) -o main $(LDFLAGS)

clean:
	rm -f knn_secuencial testing main

