# README

## 新版本特性 v2.3.0
1. 支持对全局、session、graph 的options进行添加，从而进一步调优模型，相关案例请参考 [性能调优方法论](#性能调优方法论)。   
2. 支持采用分档模式将符合条件模型转化为静态图，提高吞吐性能。
3. 补充如何采用分档模式+锁核+调整精度，进一步提高性能。
* 在AscendHub下载镜像时需确认好版本，若使用旧版本镜像，需要手工下载源码编译生成新的backend后才能支持新特性。

## 版本特性 v2.2.0
1. 支持从onnx文件读取模型输入输出信息, config中若无指定input，output，将会自动从文件中读取；   
2. 调整动态图在多实例下使用多Session方式，提高并发吞吐(显存占用会增高)；
3. 支持动态batch场景小batch动态合并特性，配合多Session，提高吞吐；
4. 补充调优方法论以及cnclip模型的最佳实践；
5. 支持多模型特性，可支持同时拉起多个模型，提高现存利用率；
6. 支持非0轴动态shape场景；
7. 支持TensorFlow的pb文件。

## 介绍
ge-backend基于triton inference server框架实现对接NPU生态，快速实现传统CV\NLP模型的服务化。   
triton inference server相关介绍请参考：
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html

### 实现原理
triton inference server 提供了Custom backend 接口，允许通过自定义backend实现NPU设备接入。
1.  将本工程编译的backend文件libnpu_ge.so安装到 {Triton-server源码安装目录}/backends/npu_ge/,  启动triton-inference-server服务端, server在拉起模型过程中根据模型设置，选择npu_ge后端对推理请求进行分发。  
2.  ge_backend 采用 GE组图方式进行推理，基于C++实现，支持GE的图优化、UB融合、多流并行等诸多特性，以便更好的为服务化模型提供更高吞吐。   
3.  模型在使用该框架时需要统一转换为Onnx格式，并基于triton-inference-server规范，配置模型相关config以及版本信息。

### 特性支持情况
|  特性名称 | 介绍 |支持情况 |
|  ------  | ------   |------   |
|  多模型 |可支持一个server启动多个模型| √ |
|  多实例 |模型可同时处理多个请求，此特性需搭配多流并行或多卡使用| √ |
|  多卡支持 |一个模型可同时跑在多张卡上，每张卡可配置>1 的实例| √ |
|  多卡负载均衡|多卡情况下能根据每张卡上任务数量动态分配请求 | 目前仅支持所有请求shape一致场景  |
|  动态batch|支持input、output 的0轴为可变场景| √ |
|  GE静态图 |通过shape固定，实现初始化图时分配好所有显存，提高图执行效率| √ |
|  多流并行 |多实例场景下NPU支持多Stream，提高NPU利用率| √ |
|  锁核 |配置每一条stream使用Cube以及Vector核心数量，以便多stream情况下提高吞吐| √ |
|  非0轴动态 |支持非0轴情况下的动态shape| √ * |
|  自动配置 | 支持onnx模型自动读取input、output免配置| √ * |
*  *若output中包含动态轴，在导出onnx时需指定其与input中轴的关系，详情请查看 [Torch模型转换为onnx](#执行推理)  

## 快速入门
用户可参考 [快速入门](docs/快速入门.md) 文档，从0-1掌握如何使用该backend快速接入NPU，实现小模型服务化。

## 问题定位&工具
在服务启动、推理过程中遇到问题，可参考[问题定位](docs/问题定位.md) 文档，定位具体问题

## 性能调优方法论
用户可参考此文档逐步提高模型吞吐，将性能调整至最优，文章最后以cnclip模型迁移为例，展示模型从转换至接入、运行、调优全流程，请点击 [性能调优方法论](docs/性能调优方法论.md) 查看


## 相关信息

- [贡献指南](CONTRIBUTING.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ge)

## 联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/triton-inference-server-ge-backend/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/triton-inference-server-ge-backend/discussions)参与交流。
- **微信交流群**：通过添加【GE小助手】，并反馈需要加入【triton-ge-backend交流群】，小助手会将您添加至相应交流群：
![](docs/figures/geas.jpg)