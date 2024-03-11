# import json

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart  # type: ignore


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, ptr = call[0], call[1:]
    check_cuda_err(err)
    if len(ptr) == 1:
        ptr = ptr[0]
    return ptr


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
    )


def memcpy_device_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        )
    )


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    )


class TensorRTInfer:
    """
    Implements inference for the dynamic batch size of a TensorRT engine.
    """

    def __init__(self, engine_path, max_batch_size):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)  # type: ignore
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:  # type: ignore
            assert runtime
            # meta_len = int.from_bytes(
            #     f.read(4), byteorder="little"
            # )  # read metadata length
            # metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
            # print(metadata)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = max_batch_size
            # print(np.dtype(trt.nptype(dtype)).itemsize)
            size = np.dtype(trt.nptype(dtype)).itemsize
            size *= self.batch_size
            for s in shape[1:]:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": [64] + list(shape[1:]),
                "size": size,
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def show(self):
        print(self.inputs)
        print(self.outputs)

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, inputs, batch_size):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """
        self.context.set_input_shape(
            self.inputs[0]["name"], [batch_size] + self.inputs[0]["shape"][1:]
        )

        # Prepare the output data
        output = np.zeros(
            [batch_size] + self.outputs[0]["shape"][1:], dtype=self.outputs[0]["dtype"]
        )

        # Process I/O and execute the network
        memcpy_host_to_device(
            self.inputs[0]["allocation"], np.ascontiguousarray(inputs)
        )
        self.context.execute_v2(self.allocations)
        memcpy_device_to_host(output, self.outputs[0]["allocation"])

        return output
