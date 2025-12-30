/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "utils.h"
#include "model_instance_state.h"

#define TYPE_32_BYTE 4
#define TYPE_64_BYTE 8
#define TYPE_16_BYTE 2
#define TYPE_8_BYTE 1
#define BYTE_PTR (char *)

using namespace ge;
using namespace std;
using json = nlohmann::json;

namespace triton::backend::npu_ge {

// 读取环境变量的函数
std::string ModelInstanceState::GetEnvVar(const std::string &name)
{
    const char *value = std::getenv(name.c_str());
    return (value != nullptr) ? std::string(value) : "";
}

void ModelInstanceState::LoadGeConfig(std::string &env_value, std::map<ge::AscendString, ge::AscendString> &configMap)
{
    try {
        // Parse JSON
        json jsonData = json::parse(env_value);
        // Validate JSON type
        if (!jsonData.is_object()) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("Error: JSON file content is not object format")).c_str());
            return;
        }

        // Iterate through JSON object
        for (auto &element : jsonData.items()) {
            std::string key = element.key();
            std::string value;

            // Convert value based on its type
            if (element.value().is_string()) {
                value = element.value().get<std::string>();
            } else if (element.value().is_number()) {
                value = std::to_string(element.value().get<double>());
            } else if (element.value().is_boolean()) {
                value = element.value().get<bool>() ? "true" : "false";
            } else if (element.value().is_null()) {
                value = "null";
            } else {
                // For complex types like arrays or objects, use JSON string representation
                value = element.value().dump();
            }
            configMap[ge::AscendString(key.c_str())] = ge::AscendString(value.c_str());
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("key: ") + key + " value: " + value).c_str());
        }
        return;
    } catch (const json::parse_error &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("JSON parsing error: ") + e.what()).c_str());
        return;
    } catch (const std::exception &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Error: ") + e.what()).c_str());
        return;
    }
}

// fixed-batch ir_option
void ModelInstanceState::StaticModeConfig(std::map<ge::AscendString, ge::AscendString> &parser_options)
{
    string ir_option;
    for (const auto &clientTensor : model_state_->GetInputClientTensors()) {
        ir_option += clientTensor.name_;
        ir_option += ":";
        for_each(clientTensor.dims_.begin(), clientTensor.dims_.end(), [&ir_option](int64_t dim) {
            ir_option += std::to_string(dim);
            ir_option += ",";
        });
        ir_option.back() = ';';
    }
    ir_option.pop_back();
    parser_options[ge::AscendString(ge::ir_option::INPUT_SHAPE)] = ge::AscendString(ir_option.c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("ir_option: ") + ir_option).c_str());
}

// one-batch ir_option
void ModelInstanceState::StaticModeConfigOne(std::map<ge::AscendString, ge::AscendString> &parser_options)
{
    string ir_option;
    for (const auto &clientTensor : model_state_->GetInputClientTensors()) {
        ir_option += clientTensor.name_;
        ir_option += ":1,";
        for_each(clientTensor.dims_.begin(), clientTensor.dims_.end(), [&ir_option](int64_t dim) {
            ir_option += std::to_string(dim);
            ir_option += ",";
        });
        ir_option.back() = ';';
    }
    ir_option.pop_back();
    parser_options[ge::AscendString(ge::ir_option::INPUT_SHAPE)] = ge::AscendString(ir_option.c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("ir_option: ") + ir_option).c_str());
}

// 初始化GE环境
int ModelInstanceState::InitGEEnvironment()
{
    aclError retInit = aclInit(nullptr);
    if (retInit != ACL_ERROR_NONE) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclInit failed, ret is: ") + std::to_string(retInit)).c_str());
        return RET_ERR;
    }

    std::map<ge::AscendString, ge::AscendString> ge_init_options;
    std::string ge_json(model_state_->GetRuntimeModelPath());
    ge_json += "/../ge.json";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ge_json.c_str());

    std::string env_value = GetEnvVar("GE_NPU_CONFIG");
    if (!env_value.empty()) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("GE_NPU_CONFIG = ") + env_value).c_str());
        LoadGeConfig(env_value, ge_init_options);
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, "No GE_NPU_CONFIG parameters");
        if (!model_state_->GetGeConfig().empty()) {
            for (auto &pair : model_state_->GetGeConfig()) {
                ge_init_options[ge::AscendString((pair.first).c_str())] = ge::AscendString((pair.second).c_str());
                LOG_MESSAGE(
                    TRITONSERVER_LOG_INFO,
                    (std::string("ge_init_options: ") + "key: " + pair.first + " value: " + pair.second).c_str());
            }
        }
    }

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (std::string("ge_init_options.size(): ") + std::to_string(ge_init_options.size())).c_str());

    ge::Status ret = ge::GEInitialize(ge_init_options);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string(" GEInitialize failed.") + std::to_string(ret)).c_str());
        aclFinalize();
        return RET_ERR;
    }
    return RET_OK;
}

// 初始化设备
int ModelInstanceState::InitDevices(std::vector<int> &dev_ids_)
{
    uint32_t dev_count;
    aclError retInit = aclrtGetDeviceCount(&dev_count);
    if (retInit != ACL_ERROR_NONE) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclrtGetDeviceCount failed, ret is: ") + std::to_string(retInit)).c_str());
        return RET_ERR;
    }

    if (dev_ids_.size() == 0) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("aclrtGetDeviceCount getDevice count: ") + std::to_string(dev_count)).c_str());
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, "user not set dev_ids ,use all npu card");
        for (int i = 0; i < dev_count; i++) {
            dev_ids_.push_back(i);
        }
    }

    if (dev_ids_.size() == 0) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "no npu card detected!");
        return RET_ERR;
    }
    return RET_OK;
}

// 初始化图会话
int ModelInstanceState::InitGraphSession(int dev_id, int graph_id, aclrtContext context, std::mutex &mu,
                                         ge::Session *session_)
{
    ge::Status ret;
    ge::graphStatus graph_ret;
    aclrtStream stream_ = nullptr;
    aclError aclRet;
    aclrtSetCurrentContext(context);
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("graph_id: ") + std::to_string(graph_id) + std::string("start init")).c_str());

    ge::Graph graph1;
    std::map<ge::AscendString, ge::AscendString> parser_options;

    ConfigureParserOptions(parser_options);

    if (!ParseGraph(graph_id, parser_options, graph1, mu)) {
        return RET_ERR;
    }

    if (!AddAndCompileGraph(session_, graph_id, graph1, ret)) {
        return RET_ERR;
    }

    if (!CreateAndLoadStream(session_, graph_id, stream_, ret, aclRet)) {
        return RET_ERR;
    }

    model_state_->GetScheduler()->AddInstance(graph_id, context, stream_, session_, dev_id);
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("graph_id: ") + std::to_string(graph_id) + std::string("finished")).c_str());

    return RET_OK;
}

void ModelInstanceState::ConfigureParserOptions(std::map<ge::AscendString, ge::AscendString> &parser_options)
{
    if (model_state_->GetGeStaticMode()) {
        if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL ||
            model_state_->GetDynamicBatchSupport()) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "generate one batch graph");
            StaticModeConfigOne(parser_options);
        } else {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "generate fixed batch graph");
            StaticModeConfig(parser_options);
        }
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "generate dynamic batch graph");
    }
}

bool ModelInstanceState::ParseGraph(int graph_id, std::map<ge::AscendString, ge::AscendString> &parser_options,
                                    ge::Graph &graph1, std::mutex &mu)
{
    ge::graphStatus graph_ret;
    {
        std::lock_guard<std::mutex> lock(mu);
        switch (model_state_->GetModelType()) {
            case ModelState::ModelType::ONNX:
                graph_ret = ge::aclgrphParseONNX(model_state_->GetModelPath().c_str(), parser_options, graph1);
                if (graph_ret != RET_OK) {
                    LOG_MESSAGE(
                        TRITONSERVER_LOG_ERROR,
                        (std::string("aclgrphParseONNX execute failed, ret is: ") + std::to_string(graph_ret)).c_str());
                    return false;
                }
                break;
            case ModelState::ModelType::TENSORFLOW:
                graph_ret = ge::aclgrphParseTensorFlow(model_state_->GetModelPath().c_str(), parser_options, graph1);
                if (graph_ret != RET_OK) {
                    LOG_MESSAGE(
                        TRITONSERVER_LOG_ERROR,
                        (std::string("aclgrphParseTensorFlow execute failed, ret is: ") + std::to_string(graph_ret))
                            .c_str());
                    return false;
                }
                break;
            default:
                TRITONSERVER_Error *error =
                    TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "can't find model file in model path!");
                TRITONSERVER_ErrorDelete(error);
        }
    }
    return true;
}

bool ModelInstanceState::AddAndCompileGraph(ge::Session *session_, int graph_id, ge::Graph &graph1, ge::Status &ret)
{
    ret = session_->AddGraph(graph_id, graph1);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("session_->AddGraph failed, ret is: ") + std::to_string(ret)).c_str());
        return false;
    }

    ret = session_->CompileGraph(graph_id);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("session_->CompileGraph failed, ret is: ") + std::to_string(ret)).c_str());
        return false;
    }
    return true;
}

bool ModelInstanceState::CreateAndLoadStream(ge::Session *session_, int graph_id, aclrtStream &stream_, ge::Status &ret,
                                             aclError &aclRet)
{
    // 创建一个Stream
    aclRet = aclrtCreateStream(&stream_);
    if (aclRet != ACL_SUCCESS) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclrtCreateStream execute failed, code is:") + std::to_string(aclRet)).c_str());
        return false;
    }

    std::map<AscendString, AscendString> load_graph_options = {};
    ret = session_->LoadGraph(graph_id, load_graph_options, stream_);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("session_->LoadGraph failed, ret is: ") + std::to_string(ret)).c_str());
        return false;
    }
    return true;
}

// 计算设备执行块数量
int ModelInstanceState::CalculateDeviceExecBlock(int device_count)
{
    int device_exec_block =
        (model_state_->GetTritonThreadCount() * model_state_->GetInstanceExecBlock() + device_count - 1) / device_count;
    if (model_state_->GetDeviceExecBlock() != -1) {
        device_exec_block = model_state_->GetDeviceExecBlock();
    }
    return device_exec_block;
}

// 创建设备线程
void ModelInstanceState::CreateDeviceThreads(int dev_id, int dev_index, int device_exec_block,
                                             std::vector<std::thread> &threads, std::mutex &mu)
{
    std::map<ge::AscendString, ge::AscendString> options;
    std::string str = std::to_string(dev_id);
    options[ge::AscendString("ge.session_device_id")] = ge::AscendString(str.c_str());
    options[ge::AscendString("ge.constLifecycle")] = ge::AscendString("session");

    ConfigureDumpOptions(options);

    aclrtContext context_ = nullptr;

    if (!SetDeviceAndContext(dev_id, context_)) {
        return;
    }

    std::vector<ge::Session *> sessions = {};
    CreateSessions(dev_id, dev_index, device_exec_block, options, sessions);

    CreateThreads(dev_id, dev_index, device_exec_block, mu, context_, sessions, threads);
}

void ModelInstanceState::ConfigureDumpOptions(std::map<ge::AscendString, ge::AscendString> &options)
{
    if (model_state_->GetDumpData()) {
        options[ge::AscendString("ge.exec.enableDump")] = ge::AscendString("1");
        options[ge::AscendString("ge.exec.dumpPath")] = ge::AscendString("./dump_data");
        options[ge::AscendString("ge.exec.dumpMode")] = ge::AscendString("all");
    }
}

bool ModelInstanceState::SetDeviceAndContext(int dev_id, aclrtContext &context_)
{
    aclError retInit = aclrtSetDevice(dev_id);
    if (retInit != ACL_SUCCESS) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclrtSetDevice execute failed, code is: ") + std::to_string(retInit)).c_str());
        return false;
    }

    aclrtGetCurrentContext(&context_);
    return true;
}

void ModelInstanceState::CreateSessions(int dev_id, int dev_index, int device_exec_block,
                                        const std::map<ge::AscendString, ge::AscendString> &options,
                                        std::vector<ge::Session *> &sessions)
{
    for (int j = 0; j < device_exec_block; j++) {
        int graph_id = j + device_exec_block * dev_index + model_state_->GetInitStatus() * 1000;
        ge::Session *session = nullptr;

        if (!model_state_->GetGeStaticMode()) {
            session = new Session(options);
        } else {
            if (j == 0) {
                session = new Session(options);
            } else {
                session = sessions[0];
            }
        }

        if (session == nullptr) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("Create session failed.")).c_str());
            return;
        }

        sessions.push_back(session);
    }
}

void ModelInstanceState::CreateThreads(int dev_id, int dev_index, int device_exec_block, std::mutex &mu,
                                       aclrtContext context_, const std::vector<ge::Session *> &sessions,
                                       std::vector<std::thread> &threads)
{
    for (int j = 0; j < device_exec_block; j++) {
        int graph_id = j + device_exec_block * dev_index;
        threads.emplace_back([this, &mu, graph_id, context_, dev_id, sessions, j]() {
            InitializeGraphSession(graph_id, dev_id, context_, mu, sessions[j]);
        });
    }
}

void ModelInstanceState::InitializeGraphSession(int graph_id, int dev_id, aclrtContext context_, std::mutex &mu,
                                                ge::Session *session)
{
    try {
        int result = InitGraphSession(dev_id, graph_id, context_, mu, session);
        if (result == RET_ERR) {
            init_failed_.store(true);
            RecordInitFailure(graph_id);
        }
    } catch (...) {
        init_failed_.store(true);
        RecordInitException();
    }
}

void ModelInstanceState::RecordInitFailure(int graph_id)
{
    std::lock_guard<std::mutex> lock(exception_mutex_);
    if (!init_exception_) {
        init_exception_ = std::make_exception_ptr(
            std::runtime_error("InitGraphSession failed for graph_id: " + std::to_string(graph_id)));
    }
}

void ModelInstanceState::RecordInitException()
{
    std::lock_guard<std::mutex> lock(exception_mutex_);
    if (!init_exception_) {
        init_exception_ = std::current_exception();
    }
}

// 等待所有线程完成
void ModelInstanceState::JoinAllThreads(std::vector<std::thread> &threads)
{
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

int ModelInstanceState::Init()
{
    // 使用静态 thread_local 变量，确保只初始化一次
    thread_id_ = next_id.fetch_add(1);
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ModelName: ") + model_state_->GetModelName()).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("Thread initializing... : thread id ") + std::to_string(thread_id_)).c_str());

    if (model_state_->GetInitStatus() == -1) {
        return InitializeGlobalResources();
    }

    return RET_OK;
}

int ModelInstanceState::InitializeGlobalResources()
{
    // 模型第一个 instance ,需要负责初始化
    // 将抢到的id作为标记存入模型中
    int lock = lock_id.fetch_add(1);
    model_state_->SetInitStatus(lock);
    // 整个模型中第一个抢到 1 的需要初始化模型
    if (lock == 0) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ModelName: ") + model_state_->GetModelName() +
                                               std::string(" start init env , lock_id : ") + std::to_string(lock))
                                                  .c_str());
        if (InitGEEnvironment() != RET_OK) {
            return RET_ERR;
        }
    } else {
        while (1) {
            // 按照取号顺序等待
            if (notify_id.load() == lock) {
                break;
            }
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ModelName: ") + model_state_->GetModelName() +
                                                   std::string(" wait , lock_id : ") + std::to_string(lock))
                                                      .c_str());
            sleep(1);
        }
    }

    std::vector<int> dev_ids_ = model_state_->GetDeviceIds();
    if (InitDevices(dev_ids_) != RET_OK) {
        return RET_ERR;
    }

    return InitializeDeviceThreads(dev_ids_, lock);
}

int ModelInstanceState::InitializeDeviceThreads(const std::vector<int> &dev_ids_, int lock)
{
    int device_exec_block_ = CalculateDeviceExecBlock(dev_ids_.size());
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (std::string("final device exec block count: ") + std::to_string(device_exec_block_)).c_str());

    std::vector<std::thread> threads;
    std::mutex mu;

    for (int i = 0; i < dev_ids_.size(); i++) {
        CreateDeviceThreads(dev_ids_[i], i, device_exec_block_, threads, mu);
    }

    // 等待所有线程完成
    JoinAllThreads(threads);

    // 检查是否有初始化失败
    if (init_failed_.load()) {
        HandleInitFailure();
        return RET_ERR;
    }

    // 启动调度器的批次处理线程
    inference_->Start();
    // 解锁其余模型
    notify_id.store(lock + 1);
    return RET_OK;
}

void ModelInstanceState::HandleInitFailure()
{
    if (init_exception_) {
        try {
            std::rethrow_exception(init_exception_);
        } catch (const std::exception &e) {
            std::cerr << "Init failed with exception: " << e.what() << std::endl;
        }
    }
}

TRITONSERVER_Error *ModelInstanceState::Create(ModelState *model_state,
                                               TRITONBACKEND_ModelInstance *triton_model_instance,
                                               ModelInstanceState **state)
{
    
    *state = new ModelInstanceState(model_state, triton_model_instance);
    if ((*state)->Init() != RET_OK) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNKNOWN, 
            "init instance error, please check log for more info.");
    }
    return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
    inference_ = new Inference(model_state);
}

ModelInstanceState::~ModelInstanceState()
{
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("ModelInstanceState 释放开始 ")).c_str());
    next_id.fetch_sub(1);
    if (thread_id_ == 0) {
        while (1) {
            sleep(1);
            if (next_id.load() == 0) {
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "wait sub ==0 try finalize");
                ge::GEFinalize();
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, " GEFinalize");
                aclFinalize();
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, " aclFinalize");
                break;
            }
        }
    }
    if (inference_ != nullptr) {
        delete inference_;
        inference_ = nullptr;
    }
}

int ModelInstanceState::ProcessRequests(TRITONBACKEND_Request **requests, const uint32_t request_count)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("request_count ") + to_string(request_count)).c_str());
    inference_->HandleRequest(requests, request_count);
    return RET_OK;
}

}