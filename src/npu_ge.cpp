// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include <cstdint>
#include <string>
#include "model_state.h"
#include "model_instance_state.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "acl/acl.h"
#include "onnx_parser.h"
#include "triton/backend/backend_common.h"
#include "utils.h"

namespace triton {
namespace backend {
namespace npu_ge {

extern "C" {
/// Initialize the instance of TRITONBACKEND_Backend class.
/// This function validates the API version compatibility between Triton's and MindIE's backend,
/// returning None if compatible or an error type otherwise.
/// \param backend The pointer to an instance of TRITONBACKEND_Backend class.
/// \return The result of initialization.
TRITONSERVER_Error *TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend)
{
    const char *cname;
    RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
    std::string name(cname);

    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

    // Check the backend API version that Triton supports vs. what this
    // backend was compiled against.
    uint32_t api_version_major;
    uint32_t api_version_minor;
    RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
    return nullptr;
}
TRITONSERVER_Error *TRITONBACKEND_Finalize(TRITONBACKEND_Backend *backend)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_Finalize start: ")).c_str());
    void *state = nullptr;
    LOG_IF_ERROR(TRITONBACKEND_BackendState(backend, &state), "failed to get backend state");
    return nullptr;  // success
}

/// Initialize the instance of TRITONBACKEND_Model class.
/// This function create a ModelState object and associate it with the TRITONBACKEND_Model,
/// returning None if nothing goes wrong or an error type otherwise.
/// \param model The pointer to an instance of TRITONBACKEND_Model class.
/// \return The result of initialization.
TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model)
{
    // Create a ModelState object and associate it with the
    // TRITONBACKEND_Model. If anything goes wrong with initialization
    // of the model state then an error is returned and Triton will fail
    // to load the model.
    ModelState *model_state;
    RETURN_IF_ERROR(ModelState::Create(model, &model_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void *>(model_state)));
    return nullptr;  // success
}

/// Finalize the instance of TRITONBACKEND_Model class.
/// This function clear the ModelState associated with TRITONBACKEND_Model,
/// returning None if nothing goes wrong or an error type otherwise.
/// \param model The pointer to an instance of TRITONBACKEND_Model class.
/// \return The result of finalization.
TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model)
{
    void *vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
    ModelState *model_state = reinterpret_cast<ModelState *>(vstate);
    delete model_state;

    return nullptr;  // success
}

/// Initialize the instance of TRITONBACKEND_ModelInstance class.
/// This function create a ModelInstanceState object and associate it with the TRITONBACKEND_ModelInstance,
/// returning None if nothing goes wrong or an error type otherwise.
/// \param instance The pointer to an instance of TRITONBACKEND_ModelInstance class.
/// \return The result of initialization.
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance)
{
    // Get the model state associated with this instance's model.
    TRITONBACKEND_Model *model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    void *vmodel_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodel_state));
    ModelState *model_state = reinterpret_cast<ModelState *>(vmodel_state);

    // Create a ModelInstanceState object and associate it with the
    // TRITONBACKEND_ModelInstance.
    ModelInstanceState *instance_state;
    RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void *>(instance_state)));

    return nullptr;  // success
}

/// Finalize the instance of TRITONBACKEND_ModelInstance class.
/// This function clear the ModelInstanceState associated with TRITONBACKEND_ModelInstance,
/// returning None if nothing goes wrong or an error type otherwise.
/// \param instance The pointer to an instance of TRITONBACKEND_ModelInstance class.
/// \return The result of finalization.
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance)
{
    void *vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
    ModelInstanceState *instance_state = reinterpret_cast<ModelInstanceState *>(vstate);
    delete instance_state;

    return nullptr;  // success
}

/// Serve as the sole entry point for inference execution in the backend.
/// This function converts requests from the Triton side into the MindIE request type
/// and stores them in the ModelState's task queue for cyclic reading by the LlmManager,
/// returning None if nothing goes wrong or an error type otherwise.
/// \param instance The pointer to an instance of TRITONBACKEND_ModelInstance class.
/// \param requests Head pointer of the request queue on the Triton side.
/// \param request_count Total number of requests.
/// \return The result of finalization.
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                                       TRITONBACKEND_Request **requests, const uint32_t request_count)
{
    ModelInstanceState *instance_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void **>(&instance_state)));

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("model instance ") + instance_state->Name() + ", executing " +
                                           std::to_string(request_count) + " requests")
                                              .c_str());

    int ret = instance_state->ProcessRequests(requests, request_count);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("ProcessRequests fail please check triton log find error info")).c_str());
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNKNOWN,
                                     "ProcessRequests fail please check triton log find error info");
    }
    return nullptr;  // success
}

}  // extern "C"

}
}
}  // namespace triton::backend::mindie
