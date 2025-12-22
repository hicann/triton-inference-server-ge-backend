import sys
import os
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "image"
with httpclient.InferenceServerClient("localhost:8000") as client:
    model_input = np.random.rand(1, 3, 224, 224).astype(np.float32) 
    inputs = [
        httpclient.InferInput(
            "image", model_input.shape, np_to_triton_dtype(model_input.dtype)
        )
    ]
    inputs[0].set_data_from_numpy(model_input)   

    outputs = [
        httpclient.InferRequestedOutput("unnorm_image_features")
    ]
    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("unnorm_image_features")
    print("#################")
    
    print(output0_data.shape)