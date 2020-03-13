# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import os
from post_process import yolo_post_process, letterbox_image
from tensorflow.keras.utils import multi_gpu_model

# fix a memory allocation bug, refer to https://www.tensorflow.org/guide/gpu?hl=zh-CN
tf.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# class OutputLayer(keras.layers.Layer):
#     def __init__(self, anchors, class_names, score, iou, input_image_shape):
#         super(OutputLayer, self).__init__()
#         self.anchors = anchors
#         self.class_names = class_names
#         self.score = score
#         self.iou = iou
#         self.input_image_shape = input_image_shape
#
#     def call(self, inputs):
#         return yolo_eval(inputs, self.anchors,
#                          len(self.class_names), self.input_image_shape,
#                          score_threshold=self.score, iou_threshold=self.iou)


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "classes_num": 80,
        "score": 0.6,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
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
        self.generate()

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        # loaded = tf.saved_model.load(model_path)
        # self.inference_func = loaded.signatures["serving_default"]

        # Load model
        try:
            raw_model = load_model(model_path)
        except:
            print('load model failed.')

        print('{} model loaded.'.format(model_path))

        if self.gpu_num >= 2:
            raw_model = multi_gpu_model(raw_model, gpus=self.gpu_num)
        # boxes, scores, classes = OutputLayer(self.anchors, self.class_names, self.score, self.iou, input_image_shape)(
        #     raw_model.output)
        # self.yolo_model = keras.Model(inputs=[raw_model.input, input_image_shape], outputs=[boxes, scores, classes])
        self.yolo_model = raw_model

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        input_image_size = np.array([image.size[1], image.size[0]])
        input_image_size = np.expand_dims(input_image_size, 0)

        yolo_output = self.yolo_model.predict([image_data], verbose=1)
        # with tf.device("/gpu:0"):
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
