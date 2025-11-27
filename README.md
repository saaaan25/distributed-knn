# KNN paralelo

## Preparación del entorno
Buildear la imagen
```
docker build -t distributed_knn .
```

Ejecutar el contenedor
```
docker run -it distributed_knn
```

Entrar al contenedor desde master
```
docker exec -it master bash
```

Ejecutar el Makefile
```
make
```

Para ejecutar Makefile con cambios
```
make clean
```

## Ejecución de archivos
KNN Secuencial
```
./knn_secuencial 4.3 0.4 7
```

KNN Distribuido para .txt
```
mpirun -np 4 ./testing 4.3 0.4 7
```

KNN Distribuido con .karas
```
mpirun -np 4 ./main dataset/data.karas dataset/labels.karas 7
```
