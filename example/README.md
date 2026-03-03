# Resnet example 运行教程

## 模型准备
从网站下载onnx文件:
https://media.githubusercontent.com/media/onnx/models/refs/heads/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx?download=true

在example/resnet 文件夹下创建名称为 "1" 的文件夹，并将下载的onnx文件放置此文件夹中。最终目录结构如下：   
```
example
└── resnet
    ├── 1
    │   └── resnet18-v1-7.onnx
    └── config.pbtxt
```
## 运行推理服务
尝试运行triton inference server：(建议使用AscendHub中的镜像直接运行)
``` shell
/opt/tritonserver/bin/tritonserver --model-repository {/path/to/example}
```

启动完成后，在输出中可看到相应的 http端口信息。
```
I0301 14:17:48.002634 11040 grpc_server.cc:2519] Started GRPCInferenceService at 0.0.0.0:8001
I0301 14:17:48.002913 11040 http_server.cc:4637] Started HTTPService at 0.0.0.0:8000
I0301 14:17:48.044199 11040 http_server.cc:320] Started Metrics Service at 0.0.0.0:8002
```

## 服务端调用测试
通过调用client.py 进行测试：
```
cd example
python client.py
```

执行成功后打印如下：
```
resnetv24_dense0_fwd shape (1, 1000)
resnetv24_dense0_fwd data [[-1.4480009 -0.14706227 0.71502316 0.60883063 1.0058776 1.0106554
1.0276837 -0.89346164 -0.9704908 -0.7546704 -0.4772439 0.57412636
-0.39269644 0.37755248 -0.4234915 -0.51555425 -1.4987887 -1.698892
...
```