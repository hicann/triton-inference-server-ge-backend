#ifndef INFERENCE_H
#define INFERENCE_H

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <map>
#include <stack>

#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "all_ops.h"
#include "acl/acl.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonbackend.h"
#include "scheduler.h"
#include "model_state.h"

namespace triton::backend::npu_ge {

class Inference {
public:
    Inference(ModelState *model_state);

    // 线程控制
    std::atomic<Scheduler::Status> status_{Scheduler::Status::IDLE};
    // 超时时间配置参数
    std::chrono::microseconds timeout_;

    // 工具方法
    int GetValueByKey(size_t key) const;
    ge::DataType TransToGeData(size_t key) const;

    void FreeDevResources(std::vector<void *> indev_buffer_, std::vector<void *> outdev_buffer_);
    void CalculateBatchDistribution(int batch_total, size_t instance_count, std::vector<int> &input_offset,
                                    std::vector<int> &batch_result);
    void GetExecutionBatchParams(int batch, int &exec_batch, int &cycle_count);

    int CheckInferenceResult(int success_count, int invalid_count, int total_instances);
    std::unique_ptr<int64_t[]> CreateOutputShape(size_t output_index, uint32_t &size, int batch_total);

    int Start();
    void InitTypeMappings();

    int BatchSplicTasks(std::vector<TRITONBACKEND_Request *> &batch_tasks, std::vector<int> &batch_result);

    void CalculateCombineBatchDistribution(std::vector<int> &input_offset, std::vector<int> &batch_combine);
    int BuildComblineResponse(TRITONBACKEND_Request *request, std::vector<void *> &outhost_buffer_,
                              std::vector<int64_t> &outsize, int batch_total, int &input_offset,
                              std::vector<int> &outhost_line_size_);
    int GetTotalBatch(std::vector<int> &batch_result);
    int BuildSingleCombineOutput(TRITONBACKEND_Response *response, size_t output_index, void *outhost_buffer,
                                 int64_t buffer_size, int batch_total);
    void JoinAllThreads(std::vector<std::thread> &threads);
    ModelState *model_state_ = nullptr;

    struct ModelTensorInfo {
        std::string name_;
        size_t dtype_;
        std::vector<int64_t> tensor_dims_;
    };
    std::vector<ModelTensorInfo> input_tensor_info_;
    std::vector<ModelTensorInfo> output_tensor_info_;
    void SetTensorInfo(ModelState *model_state_);
    bool IsOperator(char c);
    int GetPriority(char op);
    int Calculate(int a, int b, char op);
    int GetDimNum(std::map<std::string, int> values, std::string formula);
    void PrintOutputInfo(std::map<std::pair<size_t, size_t>, triton::backend::npu_ge::ModelState::Express> &m1);
    void PrintModelTensorInfo(std::vector<ModelTensorInfo> &tensor_info);
    void CalcNegativeOne();
    int HandleRequest(TRITONBACKEND_Request **requests, const uint32_t request_count);
    void SetBatchTaskAndResult(std::vector<TRITONBACKEND_Request *> &batch_tasks, std::vector<int> &batch_result,
                               TRITONBACKEND_Request **requests, int i);
    std::string ReplaceVariables(std::string &formula, std::map<std::string, int> &values);
    std::vector<std::string> InfixToPostfix(std::string &formula);
    int EvaluatePostfix(std::vector<std::string> &postfix);
    void PrintOutputDimensions();
    void ProcessMapEntries(std::map<std::pair<size_t, size_t>, triton::backend::npu_ge::ModelState::Express> &m1);
    void ProcessValuesWithBatchOffset(ModelState::Express &ex, std::map<std::string, int> &values1);
    void ProcessValuesWithoutBatchOffset(ModelState::Express &ex, std::map<std::string, int> &values1);
    void LogResult(std::pair<size_t, size_t> &index);
    bool CanCombine(ModelState *model_state);
    void ProcessFormulaCharacters(std::string &formula, std::stack<char> &opStack, std::vector<std::string> &output,
                                  std::string &currentNumber);
    int BuildOutputTensor(size_t output_index, int exec_batch, void *out_dev_buffer, gert::Tensor &tensor);
    void ProcessCurrentNumber(std::string &currentNumber, std::vector<std::string> &output);
    void ProcessCharacter(char c, std::stack<char> &opStack, std::vector<std::string> &output);
    void ProcessRightParenthesis(std::stack<char> &opStack, std::vector<std::string> &output);
    void ProcessOperator(char c, std::stack<char> &opStack, std::vector<std::string> &output);
    void HandleRemainingOperators(std::stack<char> &opStack, std::vector<std::string> &output);
    int PrepareOutputTensors(std::vector<int> &outhost_line_size_, int exec_batch, std::vector<void *> &outdev_buffer_,
                             std::vector<gert::Tensor> &outputs);
    int AllocateSingleMemoryV2(ModelState *model_state, size_t j, std::vector<void *> &indev_buffer_,
                               std::vector<int> &indev_line_size_, const int64_t *shape, uint32_t dims_count,
                               const void *buffer, uint64_t buffer_size);
    int ProcessRequestInputsV2(TRITONBACKEND_Request *request, size_t request_index, std::vector<int> &batch_result,
                               std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                               std::vector<int> &input_offset);
    int ProcessInputBuffersV2(std::vector<TRITONBACKEND_Request *> &batch_tasks, std::vector<int> &batch_result,
                              std::vector<int> &input_offset, std::vector<void *> &indev_buffer_,
                              std::vector<int> &indev_line_size_);
    int AllocateCombinedMemoryV2(ModelState *model_state, std::vector<void *> &indev_buffer_,
                                 std::vector<int> &indev_line_size_, int batch_total);
    int ExtractCombineInputInfoV2(int batch_total, std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                  std::vector<int> &batch_result, std::vector<int> &input_offset,
                                  std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_);
    int PrepareOutputBuffersV2(std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outsize,
                               std::vector<int> &outhost_line_size_, int batch_total);
    void FreeHostResourcesV2(std::vector<void *> outhost_buffer_);
    int BuildInputTensorV2(size_t input_index, int exec_batch, std::vector<void *> &indev_buffer_, gert::Tensor &tensor,
                           int cycle_count, std::vector<int> &indev_line_size_, std::vector<void *> &oneBtachDev);
    int PrepareInputTensorsV2(std::vector<int> &indev_line_size_, int exec_batch, int instance_index,
                              std::vector<void *> &indev_buffer_, std::vector<gert::Tensor> &inputs, int cycle_count,
                              std::vector<void *> &oneBtachDev);
    int ExecuteInferenceCycleV2(Scheduler::Instance *instance, std::vector<void *> &indev_buffer_,
                                std::vector<int> &indev_line_size_, std::vector<void *> &outhost_buffer_,
                                std::vector<int> &outhost_line_size_, std::vector<gert::Tensor> &inputs,
                                std::vector<gert::Tensor> &outputs, int exec_batch, int cycle_count, int instance_index,
                                const std::vector<int> &input_offset);
    int ExecuteSingleInferenceV2(Scheduler::Instance *instance, int instance_index, int instance_count,
                                 std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                                 std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
                                 const std::vector<int> &input_offset, const std::vector<int> &batch_result);
    int ProcessSingleInstanceV2(Scheduler::Instance *instance, int instance_index, int instance_count,
                                std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                                std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
                                const std::vector<int> &input_offset, const std::vector<int> &batch_result,
                                int &success_count, int &invalid_count);
    int ExecuteInferenceV2(std::vector<Scheduler::Instance *> instances, std::vector<void *> &indev_buffer_,
                           std::vector<int> &indev_line_size_, std::vector<void *> &outhost_buffer_,
                           std::vector<int> &outhost_line_size_, const std::vector<int> &input_offset,
                           const std::vector<int> &batch_result);
    int ExecuteInferenceParallelV2(std::vector<Scheduler::Instance *> instances, std::vector<void *> &indev_buffer_,
                                   std::vector<int> &indev_line_size_, std::vector<void *> &outhost_buffer_,
                                   std::vector<int> &outhost_line_size_, const std::vector<int> &input_offset,
                                   const std::vector<int> &batch_result);
    int ProcessCombineRequestV2(std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                std::vector<Scheduler::Instance *> &instances, std::vector<int> &batch_result);
    void FreeDevResourcesUtils(std::vector<void *> &dev_buffer_);

private:
    std::map<size_t, size_t> size_by_type_;
    std::map<size_t, ge::DataType> ge_data_trans_;
};
}

#endif