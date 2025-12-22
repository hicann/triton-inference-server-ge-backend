/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "inference.h"
#include "utils.h"

using namespace std;
namespace triton {
namespace backend {
namespace npu_ge {

#define TYPE_32_BYTE 4
#define TYPE_64_BYTE 8
#define TYPE_16_BYTE 2
#define TYPE_8_BYTE 1
#define BYTE_PTR char *
#define NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE 2
#define NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM 3
#define PRIORITY_HIGH 2
#define PRIORITY_MID 1
#define PRIORITY_LOW 0
#define OPERATOR_PAIR 2

// 初始化类型映射
void Inference::InitTypeMappings()
{
    size_by_type_ = {{TRITONSERVER_TYPE_BOOL, TYPE_8_BYTE},    {TRITONSERVER_TYPE_UINT8, TYPE_8_BYTE},
                     {TRITONSERVER_TYPE_UINT16, TYPE_16_BYTE}, {TRITONSERVER_TYPE_UINT32, TYPE_32_BYTE},
                     {TRITONSERVER_TYPE_UINT64, TYPE_64_BYTE}, {TRITONSERVER_TYPE_INT8, TYPE_8_BYTE},
                     {TRITONSERVER_TYPE_INT16, TYPE_16_BYTE},  {TRITONSERVER_TYPE_INT32, TYPE_32_BYTE},
                     {TRITONSERVER_TYPE_INT64, TYPE_64_BYTE},  {TRITONSERVER_TYPE_FP16, TYPE_16_BYTE},
                     {TRITONSERVER_TYPE_FP32, TYPE_32_BYTE},   {TRITONSERVER_TYPE_FP64, TYPE_64_BYTE},
                     {TRITONSERVER_TYPE_BYTES, TYPE_8_BYTE},   {TRITONSERVER_TYPE_BF16, TYPE_16_BYTE}};

    ge_data_trans_ = {{TRITONSERVER_TYPE_BOOL, ge::DT_BOOL},     {TRITONSERVER_TYPE_UINT8, ge::DT_UINT8},
                      {TRITONSERVER_TYPE_UINT16, ge::DT_UINT16}, {TRITONSERVER_TYPE_UINT32, ge::DT_UINT32},
                      {TRITONSERVER_TYPE_UINT64, ge::DT_UINT64}, {TRITONSERVER_TYPE_INT8, ge::DT_INT8},
                      {TRITONSERVER_TYPE_INT16, ge::DT_INT16},   {TRITONSERVER_TYPE_INT32, ge::DT_INT32},
                      {TRITONSERVER_TYPE_INT64, ge::DT_INT64},   {TRITONSERVER_TYPE_FP16, ge::DT_FLOAT16},
                      {TRITONSERVER_TYPE_FP32, ge::DT_FLOAT},    {TRITONSERVER_TYPE_BYTES, ge::DT_STRING},
                      {TRITONSERVER_TYPE_BF16, ge::DT_BF16}};
}

void Inference::PrintModelTensorInfo(std::vector<ModelTensorInfo> &tensor_info)
{
    for (size_t j = 0; j < tensor_info.size(); j++) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("name ") + tensor_info[j].name_).c_str());
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("dtype_ ") + to_string(static_cast<int>(tensor_info[j].dtype_))).c_str());
        for (size_t i = 0; i < tensor_info[j].tensor_dims_.size(); i++) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("dim") + to_string(static_cast<int>(i)) + " " +
                                                   to_string(static_cast<int>(tensor_info[j].tensor_dims_[i])))
                                                      .c_str());
        }
    }
}

void Inference::SetTensorInfo(ModelState *model_state_)
{
    input_tensor_info_.clear();
    output_tensor_info_.clear();
    vector<int64_t> &preferBatch = model_state_->GetDynamicmode().preferred_batch_sizes;
    int input_num = model_state_->GetInputCount();
    int output_num = model_state_->GetOutputCount();
    for (int i = 0; i < input_num; i++) {
        auto tensor = model_state_->GetInputClientTensors()[i];
        ModelTensorInfo modelTensorInfo;
        modelTensorInfo.name_ = tensor.name_;
        modelTensorInfo.dtype_ = tensor.dtype_;
        if (preferBatch.size() > 0 && (model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH)) {
            modelTensorInfo.tensor_dims_ = tensor.dims_;
        }
        input_tensor_info_.push_back(modelTensorInfo);
    }
    for (int i = 0; i < output_num; i++) {
        auto tensor = model_state_->GetOutputClientTensors()[i];
        ModelTensorInfo modelTensorInfo;
        modelTensorInfo.name_ = tensor.name_;
        modelTensorInfo.dtype_ = tensor.dtype_;
        output_tensor_info_.push_back(modelTensorInfo);
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ModelTensorInfo: ")).c_str());
    PrintModelTensorInfo(input_tensor_info_);
    PrintModelTensorInfo(output_tensor_info_);
}

int Inference::GetTotalBatch(std::vector<int> &batch_result)
{
    int batch_total = 0;
    for (auto &batch : batch_result) {
        batch_total += batch;
    }
    return batch_total;
}

Inference::Inference(ModelState *model_state)
{
    InitTypeMappings();
    model_state_ = model_state;
    SetTensorInfo(model_state_);
}

int Inference::GetValueByKey(size_t key) const
{
    auto it = size_by_type_.find(key);
    if (it != size_by_type_.end()) {
        return it->second;
    } else {
        std::cerr << "键 " << key << " 不存在" << std::endl;
        return RET_ERR;
    }
}

int Inference::Start()
{
    if (model_state_ == nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("modelState_ == nullptr Start fail")).c_str());
        return RET_ERR;
    }
    if (status_ == Scheduler::Status::RUNNING) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Dynamicmode param already set up")).c_str());
        return RET_OK;
    }

    int64_t delaytime = model_state_->GetDynamicmode().max_queue_delay_microseconds_;
    timeout_ = std::chrono::microseconds(delaytime);

    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("timeout_ ") + std::to_string(delaytime)).c_str());

    vector<int64_t> &preferBatch = model_state_->GetDynamicmode().preferred_batch_sizes;
    for (size_t i = 0; i < preferBatch.size(); i++) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("preferBatch") + std::to_string(i) + " " + std::to_string(preferBatch[i])).c_str());
    }

    status_ = Scheduler::Status::RUNNING;
    return RET_OK;
}

// 组合批次后计算分配
void Inference::CalculateCombineBatchDistribution(std::vector<int> &input_offset, std::vector<int> &batch_combine)
{
    input_offset.push_back(0);
    for (size_t i = 1; i < batch_combine.size(); ++i) {
        int temp = input_offset.back() + batch_combine[i - 1];
        input_offset.push_back(temp);
    }
}

// 构建输入信息
int Inference::ExtractCombineInputInfo(int batch_total, std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                       std::vector<int> &batch_result, std::vector<void *> &inhost_buffer_,
                                       std::vector<int> &inhost_line_size_, std::vector<int> &input_offset)
{
    aclError acl_ret;
    // 计算每个请求的起始复制位置
    CalculateCombineBatchDistribution(input_offset, batch_result);
    // 能组合的情况 提前申请内存
    if (model_state_->GetDynamicmode().preferred_batch_sizes.size() > 0 &&
        model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH && model_state_->GetInToOutMap().size() == 0) {
        // 先申请总内存 这个时候的input_tensor_info_[j].tensor_dims_ 从配置文件中读 因为输入不存在未知批次
        for (size_t i = 0; i < model_state_->GetInputCount(); i++) {
            size_t sample_element_size =
                GetValueByKey(model_state_->GetInputClientTensors()[i].dtype_);  // 单个元素大小
            int64_t total_out_size = static_cast<int64_t>(sample_element_size);
            int per_buffer_size = sample_element_size;
            if (input_tensor_info_[i].tensor_dims_.size() == 0) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                            (std::string("input_tensor_info_[i].tensor_dims_.size() is: empty")).c_str());
                return RET_ERR;
            }
            vector<int64_t> &dims_in = input_tensor_info_[i].tensor_dims_;
            total_out_size *= batch_total;
            for (size_t k = 0; k < dims_in.size(); k++) {
                total_out_size *= dims_in[k];
                per_buffer_size *= dims_in[k];
            }
            void *inhost_buffer = nullptr;
            acl_ret = aclrtMallocHost(&inhost_buffer, total_out_size);
            if (acl_ret != ACL_SUCCESS) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                            (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
                return RET_ERR;
            }
            inhost_buffer_.push_back(inhost_buffer);
            inhost_line_size_.push_back(per_buffer_size);
        }
    } else {
        // 多个请求无法合并 此时只处理一个请求
        // dims_in需要在解析出传递数据的时候申请
    }
    for (size_t i = 0; i < batch_tasks.size(); i++) {
        TRITONBACKEND_Request *request = batch_tasks[i];
        for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
            TRITONBACKEND_Input *input;
            TRITONBACKEND_RequestInputByIndex(request, j, &input);
            TRITONSERVER_DataType datatype;
            const int64_t *shape;
            uint32_t dims_count;
            const char *input_name;
            TRITONBACKEND_InputProperties(input, &input_name, &datatype, &shape, &dims_count, nullptr, nullptr);
            const void *buffer;
            uint64_t buffer_size;
            TRITONSERVER_MemoryType memory_type;
            int64_t memory_type_id;
            TRITONBACKEND_InputBuffer(input, 0, &buffer, &buffer_size, &memory_type, &memory_type_id);
            // 此处向总申请的输入host内存复制

            if (model_state_->GetDynamicmode().preferred_batch_sizes.size() > 0 &&
                model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH &&
                model_state_->GetInToOutMap().size() == 0) {
                if (batch_result[i] * inhost_line_size_[j] != buffer_size) {
                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("buffer_size get error")).c_str());
                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("batch_result[") + std::to_string(i) +
                                                         "]: " + std::to_string(batch_result[i]))
                                                            .c_str());

                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("inhost_line_size_[") + std::to_string(j) +
                                                         "]: " + std::to_string(inhost_line_size_[j]))
                                                            .c_str());

                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                                (std::string("buffer_size: ") + std::to_string(buffer_size)).c_str());
                    return RET_ERR;
                }
                std::copy((BYTE_PTR)buffer, (BYTE_PTR)buffer + buffer_size,
                          (BYTE_PTR)inhost_buffer_[j] + input_offset[i] * inhost_line_size_[j]);
            } else {
                vector<int64_t> v1;
                for (size_t i = 0; i < dims_count; i++) {
                    v1.push_back(shape[i]);
                }
                input_tensor_info_[j].tensor_dims_ = v1;
                void *host_buffer = nullptr;
                acl_ret = aclrtMallocHost(&host_buffer, buffer_size);
                if (acl_ret != ACL_SUCCESS) {
                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                                (std::string("malloc host buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
                    return RET_ERR;
                }
                inhost_buffer_.push_back(host_buffer);
                std::copy((BYTE_PTR) buffer, (BYTE_PTR) buffer + buffer_size, (BYTE_PTR) host_buffer);
                size_t sample_element_size = GetValueByKey(model_state_->GetInputClientTensors()[j].dtype_);
                int per_buffer_size = sample_element_size;

                for (size_t k = 1; k < dims_count; k++) {
                    per_buffer_size *= shape[k];
                }
                inhost_line_size_.push_back(per_buffer_size);
            }
        }
    }
    return RET_OK;
}

// 准备输出缓冲区
int Inference::PrepareOutputBuffers(std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outsize,
                                    std::vector<int> &outhost_line_size_, int batch_total)
{
    aclError acl_ret;

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        size_t sample_element_size = GetValueByKey(model_state_->GetOutputClientTensors()[j].dtype_);
        int64_t total_out_size = static_cast<int64_t>(sample_element_size);
        vector<int64_t> dims_out = output_tensor_info_[j].tensor_dims_;
        if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL ||
            model_state_->GetDynamicBatchSupport()) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("use batch_total: ") + to_string(batch_total)).c_str());
            total_out_size *= batch_total;
        }

        for (auto dim : dims_out) {
            total_out_size *= dim;
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("out dim: ") + to_string(dim)).c_str());
        }

        void *outhost_buffer = nullptr;
        acl_ret = aclrtMallocHost(&outhost_buffer, total_out_size);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        outhost_buffer_.push_back(outhost_buffer);
        outsize.push_back(total_out_size);
        int index = 0;
        int modelmode = static_cast<int>(model_state_->GetModelNode());
        if (modelmode == NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE ||
            modelmode == NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM) {
            index = 1;
        }
        int per_buffer_size = sample_element_size;
        for (size_t k = index; k < dims_out.size(); k++) {
            per_buffer_size *= dims_out[k];
        }
        outhost_line_size_.push_back(per_buffer_size);
    }
    return RET_OK;
}

// 获取执行批次参数
void Inference::GetExecutionBatchParams(int batch, int &exec_batch, int &cycle_count)
{
    if (model_state_->GetGeStaticMode()) {
        if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL ||
            model_state_->GetDynamicBatchSupport()) {
            exec_batch = 1;
            cycle_count = batch;
        } else {
            exec_batch = batch;
            cycle_count = 1;
        }
    } else {
        exec_batch = batch;
        cycle_count = 1;
    }
}

ge::DataType Inference::TransToGeData(size_t key) const
{
    auto it = ge_data_trans_.find(key);
    if (it != ge_data_trans_.end()) {
        return it->second;
    } else {
        std::cerr << "键 " << key << " 不存在" << std::endl;
        return ge::DT_MAX;
    }
}

// 构建输入张量
int Inference::BuildInputTensor(size_t input_index, int exec_batch, void *dev_buffer, gert::Tensor &tensor)
{
    std::initializer_list<int64_t> origin_shape_list;
    std::initializer_list<int64_t> storage_shape_list;
    gert::StorageShape input_shape(origin_shape_list, storage_shape_list);
    gert::Shape &origin_shape = input_shape.MutableOriginShape();
    gert::Shape &storage_shape = input_shape.MutableStorageShape();

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL || model_state_->GetDynamicBatchSupport()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("input exec_batch append dim is ") + std::to_string(exec_batch)).c_str());
        origin_shape.AppendDim(exec_batch);
        storage_shape.AppendDim(exec_batch);
    }
    vector<int64_t> dims_in;
    int index = 1;
    int modelmode = static_cast<int>(model_state_->GetModelNode());
    if (modelmode == NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE ||
        modelmode == NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM) {
        index = 0;
    }
    if (model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH && model_state_->GetInToOutMap().size() == 0) {
        index = 1;
    }
    vector<int64_t> &preferBatch = model_state_->GetDynamicmode().preferred_batch_sizes;
    if (preferBatch.size() > 0 &&
        (model_state_->GetModelNode() ==
         ModelState::ModelMode::MAX_BATCH)) {  // 批处理情况下 输入维度从配置文件设置 非批处理 从实际接受张量维度设置
        index = 0;
    }
    for (size_t i = index; i < input_tensor_info_[input_index].tensor_dims_.size(); i++) {
        dims_in.push_back(input_tensor_info_[input_index].tensor_dims_[i]);
    }
    for (size_t i = 0; i < dims_in.size(); i++) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("input append dim is ") + std::to_string(dims_in[i])).c_str());
        origin_shape.AppendDim(dims_in[i]);
        storage_shape.AppendDim(dims_in[i]);
    }

    ge::DataType input_dtype = TransToGeData(model_state_->GetInputClientTensors()[input_index].dtype_);
    if (input_dtype == ge::DT_MAX) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("TransToGeData failed, ret is:") +
                                             std::to_string(model_state_->GetInputClientTensors()[input_index].dtype_))
                                                .c_str());
        return RET_ERR;
    }

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("set input_dtype is ") + std::to_string(static_cast<int>(input_dtype))).c_str());

    gert::StorageFormat input_format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    tensor = {input_shape, input_format, gert::kOnDeviceHbm, input_dtype, dev_buffer};

    return RET_OK;
}

// 准备输入张量
int Inference::PrepareInputTensors(Scheduler::Instance *instance, std::vector<void *> &inhost_buffer_,
                                   std::vector<int> &inhost_line_size_, int exec_batch, int instance_index,
                                   std::vector<void *> &indev_buffer_, std::vector<gert::Tensor> &inputs)
{
    aclError acl_ret;
    inputs.resize(model_state_->GetInputCount());

    for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("current instance_index") + std::to_string(instance_index) +
                                               " : " + std::to_string(exec_batch))
                                                  .c_str());

        int buffer_size = exec_batch * inhost_line_size_[j];
        void *dev_buffer = nullptr;
        acl_ret = aclrtMalloc(&dev_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (acl_ret != ACL_ERROR_NONE) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        indev_buffer_.push_back(dev_buffer);

        // 构建输入张量
        if (BuildInputTensor(j, exec_batch, dev_buffer, inputs[j]) != RET_OK) {
            return RET_ERR;
        }
    }
    return RET_OK;
}

// 构建输出张量
int Inference::BuildOutputTensor(size_t output_index, int exec_batch, void *out_dev_buffer, gert::Tensor &tensor)
{
    std::initializer_list<int64_t> origin_shape_list_out;
    std::initializer_list<int64_t> storage_shape_listt_out;
    gert::StorageShape output_shape(origin_shape_list_out, storage_shape_listt_out);

    gert::Shape &origin_shape_out = output_shape.MutableOriginShape();
    gert::Shape &storage_shape_out = output_shape.MutableStorageShape();

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL || model_state_->GetDynamicBatchSupport()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("output exec_batch append dim is ") + std::to_string(exec_batch)).c_str());
        origin_shape_out.AppendDim(exec_batch);
        storage_shape_out.AppendDim(exec_batch);
    }

    vector<int64_t> dims_out = output_tensor_info_[output_index].tensor_dims_;
    for (auto dim : dims_out) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("output append dim is ") + std::to_string(dim)).c_str());
        origin_shape_out.AppendDim(dim);
        storage_shape_out.AppendDim(dim);
    }

    ge::DataType output_dtype = TransToGeData(model_state_->GetOutputClientTensors()[output_index].dtype_);
    if (output_dtype == ge::DT_MAX) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("TransToGeData failed, ret is:") + std::to_string(static_cast<int>(output_dtype))).c_str());
        return RET_ERR;
    }

    gert::StorageFormat output_format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    tensor = {output_shape, output_format, gert::kOnDeviceHbm, output_dtype, out_dev_buffer};

    return RET_OK;
}

// 准备输出张量
int Inference::PrepareOutputTensors(std::vector<int> &outhost_line_size_, int exec_batch,
                                    std::vector<void *> &outdev_buffer_, std::vector<gert::Tensor> &outputs)
{
    aclError acl_ret;
    outputs.resize(model_state_->GetOutputCount());

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        int total_out_size = exec_batch * outhost_line_size_[j];
        void *out_dev_buffer = nullptr;
        acl_ret = aclrtMalloc(&out_dev_buffer, total_out_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (acl_ret != ACL_ERROR_NONE) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc host buffer failed, ret is:") + std::to_string(static_cast<int>(acl_ret)))
                            .c_str());
            return RET_ERR;
        }
        outdev_buffer_.push_back(out_dev_buffer);

        // 构建输出张量
        if (BuildOutputTensor(j, exec_batch, out_dev_buffer, outputs[j]) != RET_OK) {
            return RET_ERR;
        }
    }
    return RET_OK;
}

// 执行推理循环
int Inference::ExecuteInferenceCycle(Scheduler::Instance *instance, std::vector<void *> &inhost_buffer_,
                                     std::vector<int> &inhost_line_size_, std::vector<void *> &outhost_buffer_,
                                     std::vector<int> &outhost_line_size_, std::vector<gert::Tensor> &inputs,
                                     std::vector<gert::Tensor> &outputs, int exec_batch, int cycle_count,
                                     int instance_index, const std::vector<int> &input_offset)
{
    aclError acl_ret;
    ge::Status ret;

    for (int k = 0; k < cycle_count; k++) {
        // 复制输入数据到设备
        for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
            int buffer_size = exec_batch * inhost_line_size_[j];
            void *src_ptr =
                (void *)((BYTE_PTR) inhost_buffer_[j] + (input_offset[instance_index] + k) * inhost_line_size_[j]);

            acl_ret = aclrtMemcpy(inputs[j].GetAddr(), buffer_size, src_ptr, buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
            if (acl_ret != ACL_SUCCESS) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                            (std::string("aclrtMemcpy is failed, ret code is ") + std::to_string(acl_ret)).c_str());
                return RET_ERR;
            }
        }

        // 开始推理
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "start infer!: ");

        ret = instance->session->ExecuteGraphWithStreamAsync(instance->index, instance->stream, inputs, outputs);
        if (ret != RET_OK) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("execute model failed, ret is: ") + std::to_string(ret)).c_str());
            return RET_ERR;
        }

        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("aclrtSynchronizeStream: ")).c_str());
        acl_ret = aclrtSynchronizeStream(instance->stream);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                (std::string("aclrtSynchronizeStream is failed, ret code is ") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }

        // 复制输出数据到主机
        for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
            int total_out_size = exec_batch * outhost_line_size_[j];
            void *outdevBuffer = outputs[j].GetAddr();
            void *dst_ptr =
                (void *)((BYTE_PTR) outhost_buffer_[j] + (input_offset[instance_index] + k) * outhost_line_size_[j]);

            aclrtMemcpy(dst_ptr, total_out_size, outdevBuffer, total_out_size, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }

    return RET_OK;
}

void Inference::FreeDevResources(std::vector<void *> indev_buffer_, std::vector<void *> outdev_buffer_)
{
    for (size_t i = 0; i < indev_buffer_.size(); i++) {
        aclrtFree(indev_buffer_[i]);
    }
    indev_buffer_.clear();

    for (size_t i = 0; i < outdev_buffer_.size(); i++) {
        aclrtFree(outdev_buffer_[i]);
    }
    outdev_buffer_.clear();
}

// 执行单个推理实例
int Inference::ExecuteSingleInference(Scheduler::Instance *instance, int instance_index, int instance_count,
                                      std::vector<void *> &inhost_buffer_, std::vector<int> &inhost_line_size_,
                                      std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
                                      const std::vector<int> &input_offset, const std::vector<int> &batch_result)
{
    int batch = batch_result[instance_index];
    if (batch == 0) {
        return RET_OK;  // 无需执行
    }

    int exec_batch;
    int cycle_count;
    GetExecutionBatchParams(batch, exec_batch, cycle_count);

    aclrtSetCurrentContext(instance->context);

    std::vector<void *> indev_buffer_;
    std::vector<void *> outdev_buffer_;
    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;

    // 准备输入张量
    if (PrepareInputTensors(instance, inhost_buffer_, inhost_line_size_, exec_batch, instance_index, indev_buffer_,
                            inputs) != RET_OK) {
        FreeDevResources(indev_buffer_, outdev_buffer_);
        return RET_ERR;
    }

    // 准备输出张量
    if (PrepareOutputTensors(outhost_line_size_, exec_batch, outdev_buffer_, outputs) != RET_OK) {
        FreeDevResources(indev_buffer_, outdev_buffer_);
        return RET_ERR;
    }

    // 执行推理循环
    if (ExecuteInferenceCycle(instance, inhost_buffer_, inhost_line_size_, outhost_buffer_, outhost_line_size_, inputs,
                              outputs, exec_batch, cycle_count, instance_index, input_offset) != RET_OK) {
        FreeDevResources(indev_buffer_, outdev_buffer_);
        return RET_ERR;
    }

    FreeDevResources(indev_buffer_, outdev_buffer_);
    return RET_OK;
}

// 等待所有线程完成
void Inference::JoinAllThreads(std::vector<std::thread> &threads)
{
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// 处理单个推理实例的公共方法
int Inference::ProcessSingleInstance(Scheduler::Instance *instance, int instance_index, int instance_count,
                                     std::vector<void *> &inhost_buffer_, std::vector<int> &inhost_line_size_,
                                     std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
                                     const std::vector<int> &input_offset, const std::vector<int> &batch_result,
                                     int &success_count, int &invalid_count)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("instance_index: ") + std::to_string(instance_index) +
                                           " :batch: " + std::to_string(batch_result[instance_index]))
                                              .c_str());

    int batch = batch_result[instance_index];
    if (batch == 0) {
        invalid_count++;
        return RET_OK;  // 无效实例，但不算错误
    }

    int result = ExecuteSingleInference(instance, instance_index, instance_count, inhost_buffer_, inhost_line_size_,
                                        outhost_buffer_, outhost_line_size_, input_offset, batch_result);
    if (result == RET_OK) {
        success_count++;
    }

    return result;
}

// 检查推理结果的公共方法
int Inference::CheckInferenceResult(int success_count, int invalid_count, int total_instances)
{
    if (success_count != (total_instances - invalid_count) || invalid_count == total_instances) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "infer output failed!");
        return RET_ERR;
    }
    return RET_OK;
}

// 串行执行推理（不使用线程）
int Inference::ExecuteInference(std::vector<Scheduler::Instance *> instances, std::vector<void *> &inhost_buffer_,
                                std::vector<int> &inhost_line_size_, std::vector<void *> &outhost_buffer_,
                                std::vector<int> &outhost_line_size_, const std::vector<int> &input_offset,
                                const std::vector<int> &batch_result)
{
    if (instances.empty()) {
        return RET_OK;  // 没有实例，直接返回成功
    }

    int success_count = 0;
    int invalid_count = 0;

    for (int instance_index = 0; instance_index < instances.size(); instance_index++) {
        ProcessSingleInstance(instances[instance_index], instance_index, instances.size(), inhost_buffer_,
                              inhost_line_size_, outhost_buffer_, outhost_line_size_, input_offset, batch_result,
                              success_count, invalid_count);
    }

    model_state_->GetScheduler()->SetInstanceStatus(instances, Scheduler::Status::IDLE);

    return CheckInferenceResult(success_count, invalid_count, instances.size());
}

// 并行执行推理
int Inference::ExecuteInferenceParallel(std::vector<Scheduler::Instance *> instances,
                                        std::vector<void *> &inhost_buffer_, std::vector<int> &inhost_line_size_,
                                        std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
                                        const std::vector<int> &input_offset, const std::vector<int> &batch_result)
{
    if (instances.empty()) {
        return RET_OK;  // 没有实例，直接返回成功
    }

    std::vector<std::thread> threads;
    std::mutex mu;
    std::atomic<int> success_count{0};
    std::atomic<int> invalid_count{0};

    for (int instance_index = 0; instance_index < instances.size(); instance_index++) {
        threads.emplace_back([this, &mu, &success_count, &invalid_count, instance = instances[instance_index],
                              instance_index, count = instances.size(), &inhost_buffer_, &inhost_line_size_,
                              &outhost_buffer_, &outhost_line_size_, &input_offset, &batch_result]() {
            // 使用局部变量来收集结果，然后原子更新
            int local_success = 0;
            int local_invalid = 0;

            ProcessSingleInstance(instance, instance_index, count, inhost_buffer_, inhost_line_size_, outhost_buffer_,
                                  outhost_line_size_, input_offset, batch_result, local_success, local_invalid);

            // 原子更新计数
            if (local_invalid > 0) {
                invalid_count += local_invalid;
            }
            if (local_success > 0) {
                success_count += local_success;
            }
        });
    }

    JoinAllThreads(threads);
    model_state_->GetScheduler()->SetInstanceStatus(instances, Scheduler::Status::IDLE);

    return CheckInferenceResult(success_count, invalid_count, instances.size());
}

// 构建单个输出
int Inference::BuildSingleCombineOutput(TRITONBACKEND_Response *response, size_t output_index, void *outhost_buffer,
                                        int64_t buffer_size, int batch_total)
{
    string output_name = model_state_->GetOutputClientTensors()[output_index].name_;
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("current  output_name ") + output_name).c_str());

    TRITONBACKEND_Output *output = nullptr;
    uint32_t size = static_cast<uint32_t>(model_state_->GetOutputClientTensors()[output_index].dims_.size());

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL || model_state_->GetDynamicBatchSupport()) {
        size += 1;
    }

    std::unique_ptr<int64_t[]> shape_out = CreateOutputShape(output_index, size, batch_total);
    if (!shape_out) {
        return RET_ERR;
    }

    TRITONSERVER_Error *error = TRITONBACKEND_ResponseOutput(
        response, &output, model_state_->GetOutputClientTensors()[output_index].name_.c_str(),
        static_cast<TRITONSERVER_DataType>(model_state_->GetOutputClientTensors()[output_index].dtype_),
        shape_out.get(), size);

    if (error != nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("TRITONBACKEND_ResponseOutput failed, ")).c_str());
        return RET_ERR;
    }

    void *buffer_out;
    TRITONSERVER_MemoryType memory_type_out;
    int64_t memory_type_id_out;
    error = TRITONBACKEND_OutputBuffer(output, &buffer_out, buffer_size, &memory_type_out, &memory_type_id_out);
    if (error != nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_OutputBuffer failed, ")).c_str());
        return RET_ERR;
    }

    std::copy((BYTE_PTR)outhost_buffer, (BYTE_PTR)outhost_buffer + buffer_size, (BYTE_PTR)buffer_out);
    return RET_OK;
}

int Inference::PrepareCombineOutputBuffers(std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outsize,
                                           std::vector<int> &outhost_line_size_, int batch_total)
{
    aclError acl_ret;

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        size_t sample_element_size = GetValueByKey(model_state_->GetOutputClientTensors()[j].dtype_);
        int64_t total_out_size = static_cast<int64_t>(sample_element_size);
        vector<int64_t> dims_out = model_state_->GetOutputClientTensors()[j].dims_;
        total_out_size *= batch_total;
        for (auto dim : dims_out) {
            total_out_size *= dim;
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("out dim: ") + to_string(dim)).c_str());
        }

        void *outhost_buffer = nullptr;
        acl_ret = aclrtMallocHost(&outhost_buffer, total_out_size);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        outhost_buffer_.push_back(outhost_buffer);
        outsize.push_back(total_out_size);

        int per_buffer_size = sample_element_size;
        for (size_t k = 0; k < dims_out.size(); k++) {
            per_buffer_size *= dims_out[k];
        }
        outhost_line_size_.push_back(per_buffer_size);
    }
    return RET_OK;
}

// 构建响应
int Inference::BuildComblineResponse(TRITONBACKEND_Request *request, std::vector<void *> &outhost_buffer_,
                                     std::vector<int64_t> &outsize, int batch_total, int &input_offset,
                                     std::vector<int> &outhost_line_size_)
{
    TRITONBACKEND_Response *response;
    TRITONBACKEND_ResponseNew(&response, request);

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        void *outhost_buffer = (BYTE_PTR)(outhost_buffer_[j] + input_offset * outhost_line_size_[j]);
        if (BuildSingleCombineOutput(response, j, outhost_buffer, outsize[j], batch_total) != RET_OK) {
            TRITONBACKEND_ResponseDelete(response);
            return RET_ERR;
        }
    }

    TRITONSERVER_Error *error = TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
    if (error != nullptr) {
        TRITONBACKEND_ResponseDelete(response);
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ResponseSend failed, ")).c_str());
        return RET_ERR;
    }
    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL), "failed releasing request");

    return RET_OK;
}

void Inference::FreeHostResources(std::vector<void *> inhost_buffer_, std::vector<void *> outhost_buffer_)
{
    for (size_t i = 0; i < inhost_buffer_.size(); i++) {
        aclrtFreeHost(inhost_buffer_[i]);
    }

    inhost_buffer_.clear();
    for (size_t i = 0; i < outhost_buffer_.size(); i++) {
        aclrtFreeHost(outhost_buffer_[i]);
    }
    outhost_buffer_.clear();
}

// 计算批次分配
void Inference::CalculateBatchDistribution(int batch_total, size_t instance_count, std::vector<int> &input_offset,
                                           std::vector<int> &batch_result)
{
    input_offset.resize(instance_count, 0);
    batch_result.resize(instance_count, 0);

    int base = batch_total / instance_count;
    int remainder = batch_total % instance_count;

    for (size_t i = 0; i < instance_count; ++i) {
        batch_result[i] = base;
        if (i < remainder) {
            batch_result[i] += 1;
        }
    }

    for (size_t i = 1; i < instance_count; ++i) {
        input_offset[i] = input_offset[i - 1] + batch_result[i - 1];
    }
}

int Inference::ProcessCombineRequest(std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                     std::vector<Scheduler::Instance *> &instances, std::vector<int> &batch_result)
{
    // 设置当前上下文
    aclrtContext nowContext;
    aclrtGetCurrentContext(&nowContext);
    if (nowContext != instances[0]->context) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "process request setcurrentContext ");
        aclrtSetCurrentContext(instances[0]->context);
    }
    aclError acl_ret;
    vector<int64_t> outsize;
    std::vector<void *> outhost_buffer_;
    std::vector<void *> inhost_buffer_;
    std::vector<int> inhost_line_size_;
    std::vector<int> outhost_line_size_;
    std::vector<int> input_offset;
    // 数据合并 成一个总大小的
    int batch_total = GetTotalBatch(batch_result);
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("batch_total: ") + std::to_string(batch_total)).c_str());
    // 准备输入数据
    if (ExtractCombineInputInfo(batch_total, batch_tasks, batch_result, inhost_buffer_, inhost_line_size_,
                                input_offset) != RET_OK) {
        return RET_ERR;
    }
    // 计算未知维度
    // 根据实际的维度计算输出张量中维度是-1的实际值
    CalcNegativeOne();
    // 准备输出数据
    if (PrepareOutputBuffers(outhost_buffer_, outsize, outhost_line_size_, batch_total) != RET_OK) {
        FreeHostResources(inhost_buffer_, outhost_buffer_);
        return RET_ERR;
    }
    // 将总批次的数据 分给多个instance取处理
    std::vector<int> instance_offset;
    std::vector<int> instance_batch;
    int instance_count = instances.size();
    CalculateBatchDistribution(batch_total, instance_count, instance_offset, instance_batch);
    // 推理
    if (instances.size() == 1) {
        ExecuteInference(instances, inhost_buffer_, inhost_line_size_, outhost_buffer_, outhost_line_size_,
                         input_offset, instance_batch);
    } else {
        ExecuteInferenceParallel(instances, inhost_buffer_, inhost_line_size_, outhost_buffer_, outhost_line_size_,
                                 input_offset, instance_batch);
    }
    // 拆分请求回复
    // 输出结果重新拆分
    for (size_t i = 0; i < batch_tasks.size(); i++) {
        std::vector<int64_t> single_outsize_;
        for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
            single_outsize_.push_back(batch_result[i] * outhost_line_size_[j]);
        }
        if (BuildComblineResponse(batch_tasks[i], outhost_buffer_, single_outsize_, batch_result[i], input_offset[i],
                                  outhost_line_size_) != RET_OK) {
            FreeHostResources(inhost_buffer_, outhost_buffer_);
            return RET_ERR;
        }
    }
    return RET_OK;
}

// 创建输出形状数组
std::unique_ptr<int64_t[]> Inference::CreateOutputShape(size_t output_index, uint32_t &size, int batch_total)
{
    std::unique_ptr<int64_t[]> shape_out = std::make_unique<int64_t[]>(size);
    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL || model_state_->GetDynamicBatchSupport()) {
        shape_out[0] = batch_total;
        for (size_t k = 0; k < output_tensor_info_[output_index].tensor_dims_.size(); k++) {
            shape_out[k + 1] = output_tensor_info_[output_index].tensor_dims_[k];
        }
    } else {
        for (size_t k = 0; k < size; k++) {
            shape_out[k] = output_tensor_info_[output_index].tensor_dims_[k];
        }
    }

    return shape_out;
}

bool Inference::IsOperator(char c)
{
    return c == '+' || c == '-' || c == '*' || c == '/';
}

// 辅助函数：获取运算符优先级
int Inference::GetPriority(char op)
{
    if (op == '+' || op == '-')
        return PRIORITY_MID;
    if (op == '*' || op == '/')
        return PRIORITY_HIGH;
    return PRIORITY_LOW;
}

// 辅助函数：执行运算
int Inference::Calculate(int a, int b, char op)
{
    switch (op) {
        case '+':
            return a + b;
        case '-':
            return a - b;
        case '*':
            return a * b;
        case '/':
            if (b == 0)
                throw runtime_error("Division by zero");
            return a / b;
        default:
            throw runtime_error("Invalid operator");
    }
}

// 主函数：计算表达式
int Inference::GetDimNum(std::map<std::string, int> values, std::string formula)
{
    // 步骤1：替换变量名为对应的数值
    string processedFormula = formula;

    // 遍历map，替换所有变量名
    for (const auto &pair : values) {
        string variable = pair.first;
        string value = to_string(pair.second);

        size_t pos = 0;
        while ((pos = processedFormula.find(variable, pos)) != string::npos) {
            // 检查变量名前后是否是字母或数字，确保是完整的变量名
            bool validBefore = (pos == 0 || !isalnum(processedFormula[pos - 1]));
            bool validAfter = (pos + variable.length() == processedFormula.length() ||
                               !isalnum(processedFormula[pos + variable.length()]));
            if (validBefore && validAfter) {
                processedFormula.replace(pos, variable.length(), value);
                pos += value.length();
            } else {
                pos += variable.length();
            }
        }
    }

    // 步骤2：将中缀表达式转换为后缀表达式（逆波兰表示法）
    stack<char> opStack;
    vector<string> output;
    string currentNumber;

    for (size_t i = 0; i < processedFormula.length(); i++) {
        char c = processedFormula[i];

        if (isdigit(c)) {
            // 处理数字（可能有多位）
            currentNumber += c;
        } else {
            // 如果之前有数字，先添加到输出
            if (!currentNumber.empty()) {
                output.push_back(currentNumber);
                currentNumber.clear();
            }

            if (c == '(') {
                opStack.push(c);
            } else if (c == ')') {
                // 弹出直到遇到左括号
                while (!opStack.empty() && opStack.top() != '(') {
                    output.push_back(string(1, opStack.top()));
                    opStack.pop();
                }
                if (!opStack.empty() && opStack.top() == '(') {
                    opStack.pop();  // 弹出左括号
                }
            } else if (IsOperator(c)) {
                // 处理运算符优先级
                while (!opStack.empty() && GetPriority(opStack.top()) >= GetPriority(c)) {
                    output.push_back(string(1, opStack.top()));
                    opStack.pop();
                }
                opStack.push(c);
            }
            // 忽略空格等其他字符
        }
    }

    // 处理最后一个数字
    if (!currentNumber.empty()) {
        output.push_back(currentNumber);
    }

    // 弹出栈中剩余的所有运算符
    while (!opStack.empty()) {
        output.push_back(string(1, opStack.top()));
        opStack.pop();
    }

    // 步骤3：计算后缀表达式
    stack<int> calcStack;

    for (const string &token : output) {
        if (isdigit(token[0]) || (token.length() > 1 && isdigit(token[1]))) {
            // 如果是数字，压入栈中
            calcStack.push(stoi(token));
        } else if (IsOperator(token[0])) {
            // 如果是运算符，弹出两个操作数进行计算
            if (calcStack.size() < OPERATOR_PAIR) {
                throw runtime_error("Invalid expression");
            }
            int b = calcStack.top();
            calcStack.pop();
            int a = calcStack.top();
            calcStack.pop();
            int result = Calculate(a, b, token[0]);
            calcStack.push(result);
        }
    }

    if (calcStack.size() != 1) {
        throw runtime_error("Invalid expression");
    }

    return calcStack.top();
}

void Inference::PrintOutputInfo(std::map<std::pair<size_t, size_t>, triton::backend::npu_ge::ModelState::Express> &m1)
{
    for (auto &p1 : m1) {
        const pair<size_t, size_t> &index = p1.first;
        const ModelState::Express &ex = p1.second;
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("key ") + std::to_string(index.first) + std::string(" ") +
                                               std::to_string(index.second))
                                                  .c_str());
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ex.expressName ") + ex.expressName).c_str());
        for (auto &m : ex.dimMap) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("key ") + m.first).c_str());
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("value ") + std::to_string(m.second.first) +
                                                   std::string(" ") + std::to_string(m.second.second))
                                                      .c_str());
        }
    }
}

void Inference::CalcNegativeOne()
{
    for (size_t i = 0; i < model_state_->GetOutputCount(); i++) {
        output_tensor_info_[i].tensor_dims_ = model_state_->GetOutputClientTensors()[i].dims_;
        std::string output_dims_str = "Output tensor " + std::to_string(i) + " dimensions: [";
        for (size_t j = 0; j < output_tensor_info_[i].tensor_dims_.size(); j++) {
            output_dims_str += std::to_string(output_tensor_info_[i].tensor_dims_[j]) + " ";
        }
        output_dims_str += "]";
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, output_dims_str.c_str());
    }

    std::map<std::pair<size_t, size_t>, triton::backend::npu_ge::ModelState::Express> m1 =
        model_state_->GetInToOutMap();
    if (m1.size() == 0) {
        return;
    }
    PrintOutputInfo(m1);
    for (auto &p1 : m1) {
        const pair<size_t, size_t> &index = p1.first;
        const ModelState::Express &ex = p1.second;
        map<string, int> values1;
        if (model_state_->GetModelNode() ==
            ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE_HAVE_UNKNOW_DIM) {
            for (auto pair : ex.dimMap) {
                values1[pair.first] = input_tensor_info_[pair.second.first].tensor_dims_[pair.second.second + 1];
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                            (std::string("pair.first ") + pair.first + std::string(" value") +
                             std::to_string(input_tensor_info_[pair.second.first].tensor_dims_[pair.second.second + 1]))
                                .c_str());
            }
            // Dimensions should exclude the batch dimension
            output_tensor_info_[index.first].tensor_dims_[index.second] = GetDimNum(values1, ex.expressName);
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (std::string("Result ") + std::to_string(index.first) + std::string(" ") +
                         std::to_string(index.second) +
                         std::to_string(output_tensor_info_[index.first].tensor_dims_[index.second]))
                            .c_str());
        } else {
            for (auto pair : ex.dimMap) {
                values1[pair.first] = input_tensor_info_[pair.second.first].tensor_dims_[pair.second.second];
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                            (std::string("pair.first ") + pair.first + std::string(" value") +
                             std::to_string(input_tensor_info_[pair.second.first].tensor_dims_[pair.second.second]))
                                .c_str());
            }
            output_tensor_info_[index.first].tensor_dims_[index.second] = GetDimNum(values1, ex.expressName);
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (std::string("Result is ") + std::to_string(index.first) + std::string(" ") +
                         std::to_string(index.second) +
                         std::to_string(output_tensor_info_[index.first].tensor_dims_[index.second]))
                            .c_str());
        }
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Calculation of -1 completed");

    for (size_t i = 0; i < model_state_->GetOutputCount(); i++) {
        std::string output_dims_str = "Output tensor " + std::to_string(i) + " dimensions: [";
        for (size_t j = 0; j < output_tensor_info_[i].tensor_dims_.size(); j++) {
            output_dims_str += std::to_string(output_tensor_info_[i].tensor_dims_[j]) + " ";
        }
        output_dims_str += "]";
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, output_dims_str.c_str());
    }
}

int Inference::BatchSplicTasks(std::vector<TRITONBACKEND_Request *> &batch_tasks, std::vector<int> &batch_result)
{
    std::vector<Scheduler::Instance *> instances = model_state_->GetScheduler()->GetIdleInstances(
        nullptr, model_state_->GetInstanceExecBlock(), model_state_->GetDeviceLB());
    if (ProcessCombineRequest(batch_tasks, instances, batch_result) != RET_OK) {
        return RET_ERR;
    }
    return RET_OK;
}

void Inference::SetBatchTaskAndResult(std::vector<TRITONBACKEND_Request *> &batch_tasks, std::vector<int> &batch_result,
                                      TRITONBACKEND_Request **requests, int i)
{
    batch_tasks.push_back(requests[i]);
    TRITONBACKEND_Input *input;
    TRITONBACKEND_RequestInputByIndex(requests[i], 0, &input);

    TRITONSERVER_DataType datatype;
    const int64_t *shape;
    uint32_t dims_count;
    const char *input_name;
    TRITONBACKEND_InputProperties(input, &input_name, &datatype, &shape, &dims_count, nullptr, nullptr);
    batch_result.push_back(shape[0]);
}

int Inference::HandleRequest(TRITONBACKEND_Request **requests, const uint32_t request_count)
{
    if (model_state_->GetDynamicmode().preferred_batch_sizes.size() > 0 &&
        model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH && model_state_->GetInToOutMap().size() == 0) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Batch Request Start");
        std::vector<TRITONBACKEND_Request *> batch_tasks;
        std::vector<int> batch_result;
        for (size_t i = 0; i < request_count; i++) {
            SetBatchTaskAndResult(batch_tasks, batch_result, requests, i);
        }
        if (BatchSplicTasks(batch_tasks, batch_result) != RET_OK) {
            return RET_ERR;
        }
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "single Request Start");

        for (size_t i = 0; i < request_count; i++) {
            std::vector<TRITONBACKEND_Request *> batch_tasks;
            std::vector<int> batch_result;
            SetBatchTaskAndResult(batch_tasks, batch_result, requests, i);
            if (BatchSplicTasks(batch_tasks, batch_result) != RET_OK) {
                return RET_ERR;
            }
        }
    }
    return RET_OK;
}

}
}
}