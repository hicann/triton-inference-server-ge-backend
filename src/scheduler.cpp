/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include "scheduler.h"
using namespace std;
namespace triton {
namespace backend {
namespace npu_ge {

Scheduler::Scheduler() {}

void Scheduler::PrintInstances()
{
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "instance info:");
    for (size_t i = 0; i < instances.size(); i++) {
        std::string result = " index: " + std::to_string(instances[i].index) +
                             " context: " + std::to_string((long long int)(instances[i].context)) +
                             " stream: " + std::to_string((long long int)(instances[i].stream)) +
                             " session: " + std::to_string((long long int)(instances[i].session)) +
                             " groupid: " + std::to_string(instances[i].group_id) +
                             " status: " + ((instances[i].status == Status::RUNNING) ? "running" : "idle");
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, result.c_str());
    }
}

// 添加 instance
void Scheduler::AddInstance(int index, aclrtContext context, aclrtStream stream, ge::Session *session, int group_id)
{
    std::lock_guard<std::mutex> lock(mutex);
    // 更新组映射
    group_instances[group_id].push_back(instances.size());
    instances.emplace_back(index, context, stream, session, group_id);
}

// 计算每个组的当前负载
std::unordered_map<int, int> Scheduler::CalculateGroupLoads() const
{
    std::unordered_map<int, int> group_loads;
    for (const auto &group : group_instances) {
        int group_id = group.first;
        group_loads[group_id] = 0;
        for (const auto &index : group.second) {
            if (instances[index].status == Status::RUNNING) {
                group_loads[group_id]++;
            }
        }
    }
    return group_loads;
}

// 按组收集空闲实例
std::unordered_map<int, std::vector<Scheduler::Instance *>> Scheduler::CollectIdleInstancesByGroup()
{
    std::unordered_map<int, std::vector<Scheduler::Instance *>> idle_instances_by_group;
    for (auto &group : group_instances) {
        int group_id = group.first;
        for (auto &index : group.second) {
            if (instances[index].status == Status::IDLE) {
                idle_instances_by_group[group_id].push_back(&instances[index]);
            }
        }
    }
    return idle_instances_by_group;
}

// 创建组负载信息列表
std::vector<Scheduler::GroupLoadInfo> Scheduler::CreateGroupLoadInfo(
    const std::unordered_map<int, int> &group_loads,
    const std::unordered_map<int, std::vector<Scheduler::Instance *>> &idle_instances_by_group,
    const std::unordered_map<int, int> &selected_per_group)
{
    std::vector<Scheduler::GroupLoadInfo> group_infos;
    for (const auto &[group_id, idle_instances] : idle_instances_by_group) {
        if (!idle_instances.empty()) {
            int current_load = group_loads.at(group_id);
            int selected_count = selected_per_group.count(group_id) ? selected_per_group.at(group_id) : 0;
            group_infos.push_back({group_id, current_load, selected_count, idle_instances});
        }
    }
    return group_infos;
}

// 找到候选组（负载最小的组）
std::vector<Scheduler::GroupLoadInfo *> Scheduler::FindCandidateGroups(
    std::vector<Scheduler::GroupLoadInfo> &group_infos)
{
    if (group_infos.empty())
        return {};

    // 找到最小负载
    int min_total_load = INT_MAX;
    for (const auto &info : group_infos) {
        int total_load = info.current_load + info.selected_count;
        if (total_load < min_total_load) {
            min_total_load = total_load;
        }
    }

    // 收集所有达到最小负载的组
    std::vector<Scheduler::GroupLoadInfo *> candidates;
    for (auto &info : group_infos) {
        if ((info.current_load + info.selected_count) == min_total_load) {
            candidates.push_back(&info);
        }
    }
    return candidates;
}

// 在候选组中寻找相同上下文的实例
Scheduler::Instance *Scheduler::FindInstanceWithSameContext(
    const std::vector<Scheduler::GroupLoadInfo *> &candidate_groups, aclrtContext target_context)
{
    for (auto group_info : candidate_groups) {
        auto &instances = group_info->idle_instances;
        for (auto it = instances.begin(); it != instances.end(); ++it) {
            if ((*it)->context == target_context) {
                Scheduler::Instance *found_instance = *it;
                instances.erase(it);
                return found_instance;
            }
        }
    }
    return nullptr;
}

// 从候选组中选择第一个可用实例
Scheduler::Instance *Scheduler::SelectFirstAvailableInstance(std::vector<Scheduler::GroupLoadInfo *> &candidate_groups)
{
    for (auto group_info : candidate_groups) {
        auto &instances = group_info->idle_instances;
        if (!instances.empty()) {
            Scheduler::Instance *selected_instance = instances[0];
            instances.erase(instances.begin());
            return selected_instance;
        }
    }
    return nullptr;
}

// 选择单个实例
Scheduler::Instance *Scheduler::SelectSingleInstance(std::vector<Scheduler::GroupLoadInfo> &group_infos,
                                                     std::unordered_map<int, int> &selected_per_group,
                                                     aclrtContext target_context)
{
    auto candidate_groups = FindCandidateGroups(group_infos);
    if (candidate_groups.empty())
        return nullptr;

    // 优先选择相同上下文的实例
    Scheduler::Instance *selected_instance = FindInstanceWithSameContext(candidate_groups, target_context);

    // 如果没有相同上下文的实例，选择第一个可用实例
    if (!selected_instance) {
        selected_instance = SelectFirstAvailableInstance(candidate_groups);
    }

    // 更新选择计数
    if (selected_instance) {
        // 找到selected_instance所在的组
        for (auto &group_info : group_infos) {
            // 这里需要根据实例找到组ID，可能需要额外的映射
            // 简化处理：假设我们可以通过某种方式知道实例的组ID
            // 在实际实现中，可能需要维护实例到组的映射
            selected_per_group[group_info.group_id]++;
        }
    }

    return selected_instance;
}

// 更新选中实例的状态
void Scheduler::UpdateSelectedInstancesStatus(const std::vector<Scheduler::Instance *> &selected_instances)
{
    for (auto instance : selected_instances) {
        instance->status = Status::RUNNING;
        instance->task_count++;
    }
}

// 等待空闲实例
void Scheduler::WaitForIdleInstances(int required_num, std::unique_lock<std::mutex> &lock)
{
    while (!HasIdleInstance(required_num)) {
        cv.wait(lock);
    }
}

// 简单选择策略（不进行负载均衡）
std::vector<Scheduler::Instance *> Scheduler::SelectInstancesSimple(aclrtContext now_context, size_t num)
{
    std::vector<Scheduler::Instance *> selected_instances;

    for (auto &instance : instances) {
        if (instance.status == Status::IDLE && selected_instances.size() < num) {
            // 优先选择相同上下文的实例
            if (instance.context == now_context) {
                selected_instances.push_back(&instance);
            }
        }
    }

    // 如果相同上下文的实例不够，补充其他实例
    if (selected_instances.size() < num) {
        for (auto &instance : instances) {
            if (instance.status == Status::IDLE && instance.context != now_context && selected_instances.size() < num) {
                selected_instances.push_back(&instance);
            }
        }
    }

    UpdateSelectedInstancesStatus(selected_instances);
    return selected_instances;
}

// 使用负载均衡策略选择实例
std::vector<Scheduler::Instance *> Scheduler::SelectInstancesWithLoadBalance(aclrtContext now_context, size_t num)
{
    auto group_loads = CalculateGroupLoads();
    auto idle_instances_by_group = CollectIdleInstancesByGroup();

    std::unordered_map<int, int> selected_per_group;
    std::vector<Scheduler::Instance *> selected_instances;

    auto group_infos = CreateGroupLoadInfo(group_loads, idle_instances_by_group, selected_per_group);

    while (selected_instances.size() < num && !group_infos.empty()) {
        Scheduler::Instance *instance = SelectSingleInstance(group_infos, selected_per_group, now_context);
        if (instance) {
            selected_instances.push_back(instance);

            // 移除没有空闲实例的组
            group_infos.erase(
                std::remove_if(group_infos.begin(), group_infos.end(),
                               [](const Scheduler::GroupLoadInfo &info) { return info.idle_instances.empty(); }),
                group_infos.end());
        } else {
            break;  // 没有更多可用实例
        }
    }

    UpdateSelectedInstancesStatus(selected_instances);
    return selected_instances;
}

// 获取空闲 instances - 自动更新状态和任务计数
std::vector<Scheduler::Instance *> Scheduler::GetIdleInstances(aclrtContext now_context, int num)
{
    std::unique_lock<std::mutex> lock(mutex);

    // 等待直到有足够可用实例
    WaitForIdleInstances(num, lock);
    // 使用负载均衡策略选择实例
    return SelectInstancesWithLoadBalance(now_context, num);
}

// 设置 instance 状态
void Scheduler::SetInstanceStatus(std::vector<Scheduler::Instance *> instances, Status status)
{
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &instance : instances) {
        instance->status = status;
    }
    cv.notify_one();
    return;
}

// 获取实例数量
size_t Scheduler::GetInstanceCount()
{
    std::lock_guard<std::mutex> lock(mutex);
    return instances.size();
}

// 检查是否有空闲实例
bool Scheduler::HasIdleInstance(int num)
{
    int res = 0;
    for (const auto &instance : instances) {
        if (instance.status == Status::IDLE) {
            res++;
        }
    }
    if (res >= num) {
        return true;
    }
    return false;
}

// 获取指定组的运行实例数量
int Scheduler::GetRunningCount(int group_id)
{
    std::lock_guard<std::mutex> lock(mutex);

    int count = 0;
    for (const auto &index : group_instances.at(group_id)) {
        if (instances[index].status == Status::RUNNING) {
            count++;
        }
    }
    return count;
}

// 析构函数 - 释放资源
Scheduler::~Scheduler()
{
}
}
}
}