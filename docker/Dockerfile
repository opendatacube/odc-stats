FROM mambaorg/micromamba:git-df79b72-jammy as stats-conda

USER root
COPY env.yaml /conf/
RUN micromamba create  -y -p /env -f /conf/env.yaml && \
    micromamba clean --all --yes && \
    micromamba env export -p /env --explicit


ARG MAMBA_DOCKERFILE_ACTIVATE=1
ARG UPDATE_VERSION=1
COPY requirements.txt /conf/
# required to build numexpr
# or any --no-binary
ENV CC=/env/bin/x86_64-conda_cos6-linux-gnu-gcc \
    CXX=/env/bin/x86_64-conda_cos6-linux-gnu-g++ \
    LDSHARED="/env/bin/x86_64-conda_cos6-linux-gnu-gcc -pthread -shared -B /env/compiler_compat -L/env/lib -Wl,-rpath=/env/lib -Wl,--no-as-needed"
RUN micromamba run -p /env pip install --no-cache-dir \
    --no-build-isolation -r /conf/requirements.txt

FROM ubuntu:jammy-20240212
COPY --from=stats-conda /env /env
COPY distributed.yaml  /etc/dask/

ENV GDAL_DRIVER_PATH=/env/lib/gdalplugins \ 
    PROJ_LIB=/env/share/proj \
    GDAL_DATA=/env/share/gdal \
    PATH=/env/bin:$PATH

WORKDIR /tmp

RUN odc-stats --version 
