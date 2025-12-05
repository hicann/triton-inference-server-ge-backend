FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-910b-ubuntu22.04-py3.11
WORKDIR /workspace

# 根据需要打开，若为内网环境，需要配置代理
ENV https_proxy=http://127.0.0.1:3128
ENV http_proxy=http://127.0.0.1:3128

ENV GIT_SSL_NO_VERIFY=1
ENV CMAKE_TLS_VERIFY=0 
RUN git config --global http.proxy 127.0.0.1:3128
RUN git config --global https.proxy 127.0.0.1:3128
RUN git config --global http.sslVerify false


RUN apt-get update \
&& apt install -y --no-install-recommends \
            ca-certificates \
            autoconf \
            automake \
            build-essential \
            git \
            gperf \
            libre2-dev \
            libssl-dev \
            libtool \
            libcurl4-openssl-dev \
            libb64-dev \
            libgoogle-perftools-dev \
            patchelf \
            python3-dev \
            python3-pip \
            python3-setuptools \
            rapidjson-dev \
            scons \
            software-properties-common \
            pkg-config \
            unzip \
            wget \
            zlib1g-dev \
            libarchive-dev \
            libxml2-dev \
            libnuma-dev \
            libgtest-dev \
            googletest \
            nlohmann-json3-dev

RUN apt-get remove -y cmake


#cmake 
RUN wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8.tar.gz && \
    tar -zxf cmake-3.31.8.tar.gz && \
    rm -f cmake-3.31.8.tar.gz && \
    cd cmake-3.31.8 && \
    ./bootstrap && \
    make -j$(nproc)  && \
    make install && \
    source ~/.bashrc

# anaconda
RUN wget --no-check-certificate -O anaconda3.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.10-1-Linux-aarch64.sh -U "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36" && \
    bash anaconda3.sh -b -p /usr/local/anaconda3 && \
    rm -f anaconda3.sh

# python 3.10 
RUN export PATH=/usr/local/anaconda3/bin/:/usr/local/anaconda3/condabin:$PATH && \
    conda init bash && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/ && \
    conda config --set show_channel_urls yes && \
    conda create -y -n triton python=3.10 numpy==1.23.5 requests decorator sympy scipy attrs==21.2.0 psutil && \
    echo "conda activate triton" >> ~/.bashrc && \
    conda clean -ya && \
    source ~/.bashrc 

# boost
RUN wget --no-check-certificate https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.tar.gz && \
    tar -zxvf boost-1.81.0.tar.gz  && \
    rm -f boost-1.81.0.tar.gz && \
    cd boost-1.81.0 && \
    ./bootstrap.sh --with-libraries=all && \
    ./b2 install -j$(nproc) && \
    ldconfig

# triton inference server
ENV GIT_SSL_NO_VERIFY=1
RUN cd /workspace && \
    git clone https://github.com/triton-inference-server/server.git triton_server && \
    cd triton_server && \
    git checkout r24.02

RUN cd /workspace/triton_server && \
    mkdir build && \
    cd build && \
    export CMAKE_TLS_VERIFY=NO && \
    export PATH=/usr/local/anaconda3/envs/triton/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/anaconda3/envs/triton/lib:$LD_LIBRARY_PATH && \
    CMAKE_BUILD_PARALLEL_LEVEL=8 cmake -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver \
        -DTRITON_THIRD_PARTY_REPO_TAG=r24.02 \
        -DTRITON_COMMON_REPO_TAG=r24.02  \
        -DTRITON_CORE_REPO_TAG=r24.02  \
        -DTRITON_BACKEND_REPO_TAG=r24.02  \
        -DTRITON_VERSION=1.0.0 \
        -DTRITON_ENABLE_METRICS_GPU=OFF \
        -DTRITON_ENABLE_GRPC=ON \
        -DTRITON_ENABLE_GPU=OFF \
        -DTRITON_ENABLE_ENSEMBLE=ON \
        -DTRITON_CORE_HEADERS_ONLY=OFF \
        .. && \
    make install

RUN echo "export LD_LIBRARY_PATH=/usr/local/anaconda3/envs/triton/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

# python backend 可根据需要选择
RUN cd /workspace && \
    git clone https://github.com/triton-inference-server/python_backend.git && \
    cd python_backend && \
    git checkout r24.02 && \
    mkdir build && \
    cd build && \
    export PATH=/usr/local/anaconda3/envs/triton/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/anaconda3/envs/triton/lib:$LD_LIBRARY_PATH && \
    cmake -DTRITON_ENABLE_GPU=OFF -DTRITON_BACKEND_REPO_TAG=r24.02 -DTRITON_COMMON_REPO_TAG=r24.02 -DTRITON_CORE_REPO_TAG=r24.02 -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver .. && \
    make install

COPY ./include /workspace/triton_npu_ge_backend/include/
COPY ./src /workspace/triton_npu_ge_backend/src/
COPY ./build.sh /workspace/triton_npu_ge_backend/build.sh
COPY ./CMakeLists.txt /workspace/triton_npu_ge_backend/CMakeLists.txt

RUN export TRITON_HOME_PATH=/opt/tritonserver && \
    cd /workspace/triton_npu_ge_backend && \
    export CMAKE_TLS_VERIFY=NO && \
    export PATH=/usr/local/anaconda3/envs/triton/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/anaconda3/envs/triton/lib:$LD_LIBRARY_PATH && \
    bash build.sh
