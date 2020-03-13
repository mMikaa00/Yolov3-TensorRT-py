# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorrt as trt
from tensorrt_util import tensorrt_common as common
from tensorrt_util.yolo_calibrator import YOLOEntropyCalibrator

from post_process import yolo_post_process, letterbox_image
import os

# fix a memory allocation bug, refer to https://www.tensorflow.org/guide/gpu?hl=zh-CN
tf.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# tf.compat.v1.disable_v2_behavior()


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/tensorrt_model/yolo.pb',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "calib_img_path": "./dataset/2012_train.txt",
        "classes_num": 80,
        "score": 0.6,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
        "infer_mode": "int8",
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.anchors = self._get_anchors()
        self.engine = self._build_engine()
        self.context = self.engine.create_execution_context()
        # self.context.active_optimization_profile = 0
        # self.context.set_binding_shape(0, (1, 416, 416, 3))
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _build_engine(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.pb'), 'Frozen model must be a .pb file.'

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        # trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # CLIP_PLUGIN_LIBRARY = '/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.6.0.1'
        if not os.path.isfile(self.plugin_path):
            raise IOError("\n{}\n{}\n{}\n".format(
                "Failed to load library ({}).".format(self.plugin_path),
                "Please build the Clip sample plugin.",
                "For more information, see the included README.md"
            ))

        # dll = ctypes.CDLL(self.plugin_path)
        # init = dll.initLibNvInferPlugins
        # init = dll.initLibYoloInferPlugins
        # init(None, b'')

        # plugins = trt.get_plugin_registry().plugin_creator_list
        # for plugin_creator in plugins:
        #     print(plugin_creator.name)

        engine_path = common.model_path_to_engine_path(model_path, self.infer_mode)
        if os.path.isfile(engine_path):
            with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:
            builder.max_batch_size = 1
            builder.max_workspace_size = 1 << 28
            calibration_cache = "model_data/tensorrt_model/yolo_calibration.cache"
            if self.infer_mode == 'int8':
                calib = YOLOEntropyCalibrator(self.calib_img_path, calibration_cache, self.model_image_size,
                                              batch_size=8)
                builder.int8_mode = True
                builder.int8_calibrator = calib

            # profile = builder.create_optimization_profile()
            # profile.set_shape('input_1', (1, 416, 416, 3), (1, 416, 416, 3), (1, 416, 416, 3))
            # config.add_optimization_profile(profile)

            onnx_path = common.model_path_to_onnx_path(model_path)
            with open(onnx_path, 'rb') as model:
                print(onnx_path)
                if not parser.parse(model.read()):
                    raise TypeError("Parser parse failed.")
            # network.get_input(0).shape = (1, 416, 416, 3)
            print(network.get_output(0).shape)
            print(network.get_output(1).shape)
            print(network.get_output(2).shape)

            engine = builder.build_cuda_engine(network)
            print(engine.get_binding_name(1))

            if not engine:
                raise TypeError("Build engine failed.")
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            return engine

        # with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        #     builder.max_batch_size = 1
        #     builder.max_workspace_size = 1 << 29
        #     calibration_cache = "model_data/tensorrt_model/yolo_calibration.cache"
        #     if self.infer_mode == 'int8':
        #         calib = YOLOEntropyCalibrator(self.calib_img_path, calibration_cache, self.model_image_size,
        #                                       batch_size=8)
        #         builder.int8_mode = True
        #         builder.int8_calibrator = calib
        #     output_names = ['conv2d_59/BiasAdd',
        #                     'conv2d_67/BiasAdd',
        #                     'conv2d_75/BiasAdd']
        #     import graphsurgeon as gs
        #     plugin_map = {
        #         # "input_1": gs.create_plugin_node(name="input_1", op="Placeholder", shape=(-1, 416, 416, 3),
        #         #                                  dtype=tf.float32),
        #         # "up_sampling2d_2/ResizeNearestNeighbor": gs.create_plugin_node(
        #         #     name="trt_upsampled2d_2/ResizeNearest_TRT",
        #         #     op="ResizeNearest_TRT",
        #         #     scale=2.0),
        #         # "up_sampling2d_1/ResizeNearestNeighbor": gs.create_plugin_node(
        #         #     name="trt_upsampled2d_1/ResizeNearest_TRT",
        #         #     op="ResizeNearest_TRT",
        #         #     scale=2.0)
        #     }
        #     uff_path = common.model_path_to_uff_path(model_path)
        #     if not os.path.isfile(uff_path):
        #         uff_path = common.model_to_uff(model_path, output_names, plugin_map=plugin_map)
        #     parser.register_input("input_1", (3, 416, 416))
        #     parser.register_output("conv2d_59/BiasAdd")
        #     parser.register_output("conv2d_67/BiasAdd")
        #     parser.register_output("conv2d_75/BiasAdd")
        #     parser.parse(uff_path, network)
        #     engine = builder.build_cuda_engine(network)
        #     if not engine:
        #         raise TypeError("Build engine failed.")
        #     with open(engine_path, 'wb') as f:
        #         f.write(engine.serialize())
        #     return engine

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32', order='C')

        image_data /= 255.
        # image_data = np.ascontiguousarray(np.transpose(image_data, [2, 0, 1]))
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        input_image_size = np.array([image.size[1], image.size[0]])
        input_image_size = np.expand_dims(input_image_size, 0)

        # np.copyto(inputs[0].host, image_data.transpose(0, 3, 1, 2).ravel())
        self.inputs[0].host = image_data
        # For more information on performing inference, refer to the introductory samples.
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        yolo_output = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs,
                                          outputs=self.outputs,
                                          stream=self.stream)
        yolo_output[0] = yolo_output[0].reshape(1, 13, 13, 255)
        yolo_output[1] = yolo_output[1].reshape(1, 26, 26, 255)
        yolo_output[2] = yolo_output[2].reshape(1, 52, 52, 255)

        out_boxes, out_scores, out_classes = yolo_post_process(yolo_output, self.anchors,
                                                               self.classes_num, input_image_size,
                                                               score_threshold=self.score, iou_threshold=self.iou)

        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        return out_boxes, out_scores, out_classes

    def close_session(self):
        pass