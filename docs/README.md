# 调优文档
本文档提供了一套完整模型调优步骤，详细内容如下：
## 目录说明
```bash
Triton调优/
|-- README.md
|-- figures                  # 图片目录
|-- tools                   # 调优工具目录
|   |-- DumpGE.md
|   |-- Plog.md
|   `-- Profiling.md
|-- 性能调优方法论.md        # 调优实践介绍  
`-- CN_CLIP模型优化示例.md     # CN_CLIP模型优化示例             
```

## 文档说明
为方便开发者快速查找模型调优需要的内容，可按需查看对应文档，文档内容包括：
| 文档 | 说明 |
|:------|:-----|
| [性能调优方法论](./性能调优方法论.md) | 性能调优方法论  |
| [CN_CLIP模型优化示例](./CN_CLIP模型优化示例.md) | 以cnclip为例介绍调优过程  |
| [DumpGE](./tools/DumpGE.md) | 介绍如何Dump GE图和分析  |
| [Plog](./tools/Plog.md) |  介绍如何打开plog详细日志 |
| [Profiling](./tools/Profiling.md) |  介绍如何采集Profiling数据，分析性能瓶颈 |