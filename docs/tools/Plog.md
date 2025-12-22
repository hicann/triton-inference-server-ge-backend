# Plog打印
当模型运行出错时，如果从现有日志中无法判断具体原因，需要打开plog日志查看详细原因。
# 启动方法
当前框架已集成相关能力， 支持两种启动方式。对应 **0** 和 **1** 两种配置，当填写 **0** 时，关闭；**1** 时开启。
## 通过config配置
在config.pbtxt中添加如下参数：
```json
parameters:
{
  key: "plog",
  value: {string_value: "1"}
}
```
## 通过启动参数配置
在启动命令中配置如下参数：
```bash
--backend-config=npu_ge,plog="1"
```
# 成功启动结果
打开成功后，会在输出中包含如下所示plog日志：
![](../figures/Plog_1.png)

*注： Plog 日志较多，建议启动主进程时将输出重定向至文件中。*