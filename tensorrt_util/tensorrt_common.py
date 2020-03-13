import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import uff
import os


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# Transforms model path to uff path (e.g. /a/b/c/d.pb -> /a/b/c/d.uff)
def model_path_to_uff_path(model_path):
    uff_path = os.path.splitext(model_path)[0] + ".uff"
    return uff_path


# Transforms model path to onnx path (e.g. /a/b/c/d.pb -> /a/b/c/d.uff)
def model_path_to_onnx_path(model_path):
    onnx_path = os.path.splitext(model_path)[0] + ".onnx"
    return onnx_path


# Transforms model path to engine path (e.g. /a/b/c/d.pb -> /a/b/c/d.uff)
def model_path_to_engine_path(model_path, build_type='fp32'):
    uff_path = os.path.splitext(model_path)[0] + '_' + build_type + ".engine"
    return uff_path


# Converts the TensorFlow frozen graphdef to UFF format using the UFF converter
def model_to_uff(model_path, output_names, plugin_map={}):
    # Transform graph using graphsurgeon to map unsupported TensorFlow
    # operations to appropriate TensorRT custom layer plugins
    import graphsurgeon as gs
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph.collapse_namespaces(plugin_map)
    # Save resulting graph to UFF file
    output_uff_path = model_path_to_uff_path(model_path)
    uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        output_names,
        output_filename=output_uff_path,
        text=True
    )
    return output_uff_path
