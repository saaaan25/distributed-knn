# Imagen base estable con Debian
FROM debian:latest

# Instalar herramientas necesarias
RUN apt-get update && apt-get install -y \
    mpich \
    mpich-doc \
    build-essential \
    make \
    && apt-get clean

# Directorio de trabajo dentro del contenedor
WORKDIR /project

# Copiar el proyecto completo al contenedor
COPY . /project

# Comando por defecto
CMD ["/bin/bash"]

