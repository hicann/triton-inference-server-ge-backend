# README

## 环境准备
### 拉取镜像
为了简化编译过程，建议使用官方镜像进行编译  
从ascendhub 下载cann基础镜像
``` shell
docker pull --platform=arm64 swr.cn-south-1.myhuaweiincloud.com/ascendhub/cann:8.3.rc1-910b-ubuntu22.04-py3.11
```

### 运行镜像&进入镜像
```
docker run -itd --privileged --name=triton_npu2 --net=host --shm-size=500g \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
	-v /data/:/data/ \
	-v /home/:/home/ \
    -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-910b-ubuntu22.04-py3.11 bash

docker exec -it triton_npu bash
```

### 配置代理(可选)
```
export https_proxy=http://127.0.0.1:3128
export http_proxy=http://127.0.0.1:3128

export GIT_SSL_NO_VERIFY=1
git config --global http.proxy 127.0.0.1:3128
git config --global https.proxy 127.0.0.1:3128
git config --global  http.sslVerify  false
```
### 初始化编译环境
```
mkdir -p /workspace
cd /workspace
export CMAKE_TLS_VERIFY=0 
apt-get update
apt-get install -y --no-install-recommends \
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

apt-get remove -y cmake


# cmake 
wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8.tar.gz
tar -zxf cmake-3.31.8.tar.gz
cd cmake-3.31.8
./bootstrap
make -j$(nproc) 
make install
source ~/.bashrc 


# conda
cd /workspace

## TEST /home/g00835429/Anaconda3-2024.06-1-Linux-aarch64.sh
wget --no-check-certificate -O anaconda3.sh https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-aarch64.sh
bash anaconda3.sh -b -p /usr/local/anaconda3

export PATH=/usr/local/anaconda3/bin/:/usr/local/anaconda3/condabin:$PATH
conda init bash


conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --set show_channel_urls yes


conda create -y -n triton python=3.10 numpy==1.23.5 requests decorator sympy scipy attrs==21.2.0 psutil
echo "conda activate triton" >> ~/.bashrc
conda clean -ya

source ~/.bashrc 

# boost

## TEST  /home/g00835429/triton_server/boost-1.81.0.tar.gz
wget --no-check-certificate https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.tar.gz
tar -zxvf boost-1.81.0.tar.gz 
cd boost-1.81.0
./bootstrap.sh --with-libraries=all
./b2 install -j$(nproc)
ldconfig
```

### 下载&编译triton inference server
```
git clone https://github.com/triton-inference-server/server.git triton_server
cd triton_server
git checkout r24.02
```
在 triton_server 中创建 build_npu.sh  
添加如下内容
```
#!/bin/bash
export CMAKE_TLS_VERIFY=NO 
CACHE_DIR=/workspace/triton_server/build
INCREMENTAL_COMPILATION="OFF"
build_type="Release"
while getopts "de" opt; do
case ${opt} in
    d)
    build_type="Debug"
    ;;
    e)
    INCREMENTAL_COMPILATION="ON"
    ;;
    \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
esac
done

mkdir -p $CACHE_DIR && cd $CACHE_DIR

if [ ${INCREMENTAL_COMPILATION} == "OFF" ]; then
    rm -rf *
fi

CMAKE_BUILD_PARALLEL_LEVEL=8 cmake -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver \
        -DTRITON_THIRD_PARTY_REPO_TAG=r24.02 \
        -DTRITON_COMMON_REPO_TAG=r24.02  \
        -DTRITON_CORE_REPO_TAG=r24.02  \
        -DTRITON_BACKEND_REPO_TAG=r24.02  \
        -DTRITON_VERSION=1.0.0 \
        -DTRITON_ENABLE_METRICS_GPU=OFF \
        -DTRITON_ENABLE_GRPC=OFF \
        -DTRITON_ENABLE_GPU=OFF \
        -DTRITON_ENABLE_ENSEMBLE=ON \
        -DTRITON_CORE_HEADERS_ONLY=OFF \
        ..

make install   
```
然后执行编译
```
export LD_LIBRARY_PATH=/usr/local/anaconda3/envs/triton/lib:$LD_LIBRARY_PATH
bash build_npu.sh
```


### 尝试运行 triton inference server
/opt/tritonserver/bin/tritonserver

### 安装python backend (可选)
```
cd /workspace
git clone https://github.com/triton-inference-server/python_backend.git
cd python_backend
git checkout r24.02

mkdir build
cd build
cmake -DTRITON_ENABLE_GPU=OFF \
-DTRITON_BACKEND_REPO_TAG=r24.02 \
-DTRITON_COMMON_REPO_TAG=r24.02 \
-DTRITON_CORE_REPO_TAG=r24.02 \
-DPython_EXECUTABLE=/usr/local/anaconda3/envs/triton/bin/python \
-DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver ..

make install
```




### 编译 npu ge onnx backend
```
cd {npu ge onnx backend 目录}
export TRITON_HOME_PATH=/opt/tritonserver
bash build.sh

```

## 执行推理
### 模型转换为onnx
可使用torch自带能力进行导出
```
import torch.onnx
torch.onnx.export(model,
            (image, text),
            'model.onnx',
            input_names=['image','text'],
            output_names=['unnorm_image_features',"unnorm_text_features"],
            dynamic_axes={
                'image':{0:"bs"}, 
                'text':{0:"bs"},
                'unnorm_image_features':{0:"bs"},
                'unnorm_text_features':{0:"bs"}
            },
            export_params=True,
            do_constant_folding=False,
            opset_version=14,
            verbose=True)
```
其中：
model.onnx 为生成的onnx文件名称。  
input_names 为对应的入参名称  
output_names 为对应的出参名称  
opset_version 为使用的编译版本，默认建议选择14版本。具体请参考昇腾文档。  
dynamic_axes 为动态轴，比如image 的第0轴为 batchsize，则需要声明 {0:"bs"}   
注意！！！： 当前版本仅支持0轴为动态轴， 其他轴需要根据需求调整为固定大小。  

执行完成后会在当前路径下生成 model.onnx 模型文件。  

生成的onnx文件可以使用 onnxsim 工具进行优化。
```
pip install onnxsim 
onnxsim  model.onnx new_model.onnx
```
执行完成后会输出前后节点对比。

### 尝试运行
创建一个模型文件夹类似如下结构。
```
models
└── cnclip
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```
其中 ：
cnclip 为模型名称
1 为模型版本  
model.onnx 为需要执行推理的模型文件  
config.txt 为模型描述，具体填写内容如下：  
```
name: "cnclip"
backend: "npu_ge"
max_batch_size: 128
input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 224, 224  ]
  }
]
output [
  {
    name: "unnorm_image_features"
    data_type: TYPE_FP32
    dims: [512 ]
  }
]
instance_group [{ 
  count: 1
}
]
parameters: [
{
  key: "device_ids",
  value: {string_value: "2"}
}
]

```
当前版本支持动态bs，以及全静态方式：  
1，若bs 为动态，可以通过配置 max_batch_size 限定最大batch，
input、output中第0轴不需要声明，规范跟随triton inference server。  
2，若bs为静态，则需要去掉 max_batch_size 并在input，output中声明第0轴的大小，如上例子 dims: [1，3, 224, 224  ]  

在全为静态的情况下，可以添加参数开启静态图推理：
```
parameters: [
{
  key: "static_model",
  value: {string_value: "1"}
}
]
```

模型文件夹生成后，即可执行如下命令尝试运行推理服务：
```
/opt/tritonserver/bin/tritonserver --model-repository {/path/to/models} \
--http-port=9000 --grpc-port=9002 \
--backend-config=npu_ge,ge.aicoreNum="12|10" \
--backend-config=npu_ge,profiling="dynamic" \
--backend-config=npu_ge,dump_graph="1"
```
可用参数可参考 triton inference server 相关文档。   
在官方基础上，该插件新增了如下参数： 
--backend-config=npu_ge,static_model 是否开启ge静态图，只有在shape全部为固定值时才能开启，config.pbtxt 也可以配置，只需配置一个。默认不开启。   
--backend-config=npu_ge,ge.aicoreNum 可以配置在启用静态图时，单stream使用cube、vector核数量，| 左边为cube数量，右边为vector数量，建议根据模型cv使用情况进行调整。默认关闭。  
--backend-config=npu_ge,profiling 是否开启profiling，若为ture，则采用静态采集，程序运行后立刻开始，若为dynamic，则需要使用另一线程，具体可参考昇腾文档。 开启后会在当前目录下生成profiling文件夹。默认关闭。  
--backend-config=npu_ge,dump_graph 是否dump GE图，默认关闭。   

启动完成后，在输出中可看到相应的 http端口信息。
```
I1113 03:06:28.108960 4560 grpc_server.cc:2519] Started GRPCInferenceService at 0.0.0.0:10002
I1113 03:06:28.109231 4560 http_server.cc:4637] Started HTTPService at 0.0.0.0:10001
I1113 03:06:28.150615 4560 http_server.cc:320] Started Metrics Service at 0.0.0.0:8002
```


### client 调用
工程中自带example，用户可以调用其中的client 进行服务测试。具体使用方法请参考 triton inference server 相关文档。
```
python client.py
```

调用成功后会输出output信息。

## 性能测试
可以使用NVIDIA官方工具 perf_analyzer 测试 triton-inference-server 性能。   
拉取镜像
```
export RELEASE=23.02
docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
docker run --rm --privileged -it --net=host nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```
进入容器后，即可调用工具对模型进行测试：
```
perf_analyzer -m clip \
-x 1 \
-i http \
-u localhost:9090 \
-b 1 \
--shape image:3,224,224 \
--shape text:512 \
--concurrency-range 16
```
参数如下：   
-m	设置模型名称，如 clip \
-x	设置模型版本，根据想要测试模型的版本如实设置，如 1 \
-i	设置通信协议，选择有 http|grpc \
-u	设置通信地址url，如 localhost:9090 \
-b 	设置batch size ,如果是静态shape，则不填此参数 \
--shape	指定模型输入，以输入名称区分，如 image:3,224,224 如果填写了 -b 参数，则bs维度不需要写出 \
--concurrency-range	设置并发量，可以设定为单一值，如 16；也可以通过区间加步长进行设置，如1:1:16（start:step:end）\
-v -v 	打开详细日志 

执行成功后，会打印出相应的吞吐信息。