FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/ubuntu:22.04-linuxarm64
WORKDIR /workspace

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
    make install

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
    conda clean -ya

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

# 下载cann包，根据自己芯片类型以及需要的版本进行安装
RUN cd /tmp && \
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.3.RC2/Ascend-cann-kernels-310p_8.3.RC2_linux-aarch64.run && \
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.3.RC2/Ascend-cann-toolkit_8.3.RC2_linux-aarch64.run && \
    echo y | bash ./Ascend-cann-toolkit*.run --install --quiet --install-for-all && \
    echo y | bash ./Ascend-cann-kernels*.run --install --install-for-all && \
    rm -rf /tmp/*

RUN . /usr/local/Ascend/ascend-toolkit/set_env.sh
RUN echo "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc

# onnxruntime 支持 2.1.0 版本自动获取onnx文件配置特性
RUN cd /opt && \
    wget --no-check-certificate https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-aarch64-1.23.2.tgz && \
    tar -zxvf onnxruntime-linux-aarch64-1.23.2.tgz  && \
    mv onnxruntime-linux-aarch64-1.23.2 onnxruntime && \
    rm -f onnxruntime-linux-aarch64-1.23.2.tgz && \
    export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH && \
    echo "export LD_LIBRARY_PATH=/opt/onnxruntime/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    ldconfig

COPY ./include /workspace/triton_npu_ge_backend/include/
COPY ./src /workspace/triton_npu_ge_backend/src/
COPY ./build.sh /workspace/triton_npu_ge_backend/build.sh
COPY ./CMakeLists.txt /workspace/triton_npu_ge_backend/CMakeLists.txt

# 编译 ge backend
RUN export TRITON_HOME_PATH=/opt/tritonserver && \
    echo "export TRITON_HOME_PATH=/opt/tritonserver" >> ~/.bashrc && \
    cd /workspace/triton_npu_ge_backend && \
    export CMAKE_TLS_VERIFY=NO && \
    export PATH=/usr/local/anaconda3/envs/triton/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/anaconda3/envs/triton/lib:$LD_LIBRARY_PATH && \
    bash build.sh

# 删除构建包
RUN cd /workspace && \
    rm -rf boost-1.81.0 && \
    rm -rf cmake-3.31.8 && \
    rm -rf python_backend && \
    rm -rf triton_npu_ge_backend && \
    rm -rf triton_server
