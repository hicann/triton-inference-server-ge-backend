# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

开发者贡献场景主要包括：

## 一、贡献新特性

  若开发者基于此开源仓完成了新特性适配，或者发现BUG并修复，可参考如下流程进行PR提交：

### 1. 创建Issue需求

新建 `Requirement|需求建议` 类Issue，并阐明新增算子的设计方案。Issue一般需包含以下内容：

- **背景信息** 
- **价值/作用** 
- **设计方案**

请在提交的Issue中评论`/assign @yourself` 认领该任务。

### 2. PR提交

PR上库要求：

- 代码交付件：需提供相关特性代码，并在合理位置添加详细注释。
- 开发者自测：需提供相关自测截图，证明新特性完整性。
- 文档交付件：在当前基础上补充使用说明等内容，确保新特性被感知。
- 合规检查：
  - 代码是否符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》
  - 代码是否编译通过
  - Markdown文档语法是否符合规范
- PR提交：通过`git`命令提交目标分支PR，检查PR标题是否清晰、PR描述是否规范（指明更改内容和原因、是否关联对应Issue）、是否签署CLA。

### 3. CI门禁

通过评论 `compile` 指令触发开源仓门禁，并依据CI检测结果进行修改，目前CI门禁包含以下检查项：

- 代码编译
- 静态检查（如涉及codecheck误报，请提交给sig成员屏蔽）

门禁通过后，请在关联的Issue中@指派的Committer。

### 4. Committer检视

Committer检视后将反馈检视意见，请根据意见修改，完成后@指派的Committer。

### 5. Maintainer合入

Committer检视通过后，标注 `/lgtm`标签。Maintainer将在1天内进行最终审核，确认无问题后，将标注 `/approve` 标签合入PR。

## 二、Bug修复

如果您在本项目中发现了某些算子Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

## 三、文档纠错

如果您在本项目中发现某些算子文档描述错误，欢迎您新建Issue进行反馈和修复。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您纠正对应文档描述。

## 四、帮助解决他人Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。