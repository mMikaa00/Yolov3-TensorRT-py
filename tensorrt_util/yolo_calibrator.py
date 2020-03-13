import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from post_process import letterbox_image
import numpy as np


def data_generator(annotation_lines, batch_size, input_shape):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    count = 0
    i = 0
    while True:
        image_data = []
        for b in range(batch_size):
            count += 1
            if count > 400:
                return None
            line = annotation_lines[i].split()
            image = Image.open(line[0])
            boxed_image = letterbox_image(image, input_shape)
            image_np = np.array(boxed_image, dtype='float32', order='C')/255.
            # image_np = np.transpose(image_np, [2, 0, 1])
            image_data.append(image_np)
            i = (i+1) % n
        print("Calib count: ", count)
        yield np.ascontiguousarray(image_data)


class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_path, cache_file, input_shape, batch_size=64):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.

        self.batch_size = batch_size
        self.current_index = 0
        self.input_shape = input_shape

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(input_shape[0] * input_shape[1] * 3 * 4 * self.batch_size)
        with open(data_path) as f:
            self.lines = f.readlines()
        self.batches = data_generator(self.lines, self.batch_size, self.input_shape)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            data = next(self.batches)
            # Assume that self.device_input is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
