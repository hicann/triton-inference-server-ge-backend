/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef SCHEDULER_H
#define SCHEDULER_H

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
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "acl/acl.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonbackend.h"

namespace triton::backend::npu_ge {

class Scheduler {
public:
    enum class Status { IDLE, RUNNING };

    struct Instance {
        int index;
        aclrtContext context = nullptr;
        aclrtStream stream = nullptr;

        ge::Session *session = nullptr;

        int group_id;
        Status status;
        int task_count;  // 历史执行任务数量

        Instance(int idx, aclrtContext ctx, aclrtStream stre, ge::Session *sess, int gid)
            : index(idx),
              context(ctx),
              stream(stre),
              session(sess),
              group_id(gid),
              status(Status::IDLE),
              task_count(0)
        {
        }
    };

    // 负载均衡策略枚举
    enum class LoadBalanceStrategy {
        ROUND_ROBIN,   // 轮询
        LEAST_LOADED,  // 最少负载
        SAME_CONTEXT   // 相同上下文优先
    };

    // 组负载信息结构
    struct GroupLoadInfo {
        int group_id;
        int current_load;
        int selected_count;
        std::vector<Scheduler::Instance *> idle_instances;

        GroupLoadInfo(int id, int load, int selected, std::vector<Instance *> instances)
            : group_id(id),
              current_load(load),
              selected_count(selected),
              idle_instances(std::move(instances))
        {
        }
    };

    Scheduler();

    // 析构函数 - 释放资源
    virtual ~Scheduler();

    void PrintInstances();

    // 添加 instance
    void AddInstance(int index, aclrtContext context, aclrtStream stream, ge::Session *session, int group_id);

    // 设置 instance 状态
    void SetInstanceStatus(std::vector<Instance *> instance, Status status);

    // 获取实例数量
    size_t GetInstanceCount();

    // 检查是否有空闲实例
    bool HasIdleInstance(int num);

    // 获取指定组的运行实例数量
    int GetRunningCount(int group_id);

    // 获取空闲 instance - 自动更新状态和任务计数
    std::vector<Instance *> GetIdleInstances(aclrtContext now_context, int num);
    int batch_accu_ = 0;
    std::vector<int> batch_result_;
    std::queue<TRITONBACKEND_Request *> input_queue_;
    mutable std::mutex input_mutex_;
    std::condition_variable input_cv_;

private:
    // 负载计算相关方法
    std::unordered_map<int, int> CalculateGroupLoads() const;
    std::unordered_map<int, std::vector<Instance *>> CollectIdleInstancesByGroup();
    std::vector<GroupLoadInfo> CreateGroupLoadInfo(
        const std::unordered_map<int, int> &group_loads,
        const std::unordered_map<int, std::vector<Instance *>> &idle_instances_by_group,
        const std::unordered_map<int, int> &selected_per_group);

    // 实例选择相关方法
    std::vector<GroupLoadInfo *> FindCandidateGroups(std::vector<GroupLoadInfo> &group_infos);
    Instance *FindInstanceWithSameContext(const std::vector<GroupLoadInfo *> &candidate_groups,
                                          aclrtContext target_context);
    Instance *SelectFirstAvailableInstance(std::vector<GroupLoadInfo *> &candidate_groups);
    Instance *SelectSingleInstance(std::vector<GroupLoadInfo> &group_infos,
                                   std::unordered_map<int, int> &selected_per_group, aclrtContext target_context);

    // 选择策略方法
    void WaitForIdleInstances(int required_num, std::unique_lock<std::mutex> &lock);
    std::vector<Instance *> SelectInstancesSimple(aclrtContext now_context, size_t num);
    std::vector<Instance *> SelectInstancesWithLoadBalance(aclrtContext now_context, size_t num);

    // 状态更新方法
    void UpdateSelectedInstancesStatus(const std::vector<Instance *> &selected_instances);

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::vector<Instance> instances;
    std::unordered_map<int, std::vector<int>> group_instances;
};

}

#endif
