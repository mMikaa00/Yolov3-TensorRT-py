# Yolov3 on TensorRT7.0 and Tensorflow2.0

This repository contains Yolov3 inference on tensorrt7.0 and tensorflow 2.0.
Model is based on [darknet](https://pjreddie.com/darknet/yolo/), adapt it to 
keras and tensorrt platform.

### Test environments
    ubuntu 16.04
    Tensorflow 1.4,1.5 and 2.0
    TensorRT 7.0
    Nvidia-driver 410.78
    cuda 10.0
    cudnn 7.6.5
    python 3.5.2
    
optional:

    keras2onnx 1.6
    onnx 1.6
    
### Models
- Keras moel  
Keras model is borrowed from [keras-yolo3](https://github.com/qqwweee/keras-yolo3), which 
contains a detailed description about how to generate .h5 model. 
- TensorRT engine  
TensorRT engine are generated based on Keras model. In this case, Keras model is 
converted to onnx format and then used to generate TensorRT engine.  

Keras model can be converted to onnx as:

    python3 model_data/convert_model.py --model_path=your_dir/model.h5 --output_path=output_dir --type=onnx

then specify model path in yolo_tensorrt.py or yolo_test.py, TensorRT engine will be
generated after running yolo_test.py.

For int8 engine, calibrate images should be prepared and specify images path in yolo_tensorrt.py

- Download engine  
Or Download engine directly(waiting for upload).

### Run test

    python3 yolo_test.py --live --platform=tensorrt

more detail refer to :

    python3 yolo_test.py --help
    
### Evaluate result
int8 calibration images are 1000 pics selected in val2014

Model   | mode | dataset | MAP | MAP (0.5) | MAP(0.75)  
----  | ----  | --- | --- | --- | ---
Yolov3-416  | raw  | COCOval2014 | 0.315 | 0.561 | 0.319
Yolov3-416  | fp32 | COCOval2014 | 0.315 | 0.561 | 0.319
Yolov3-416  | int8 | COCOval2014 | 0.304 | 0.551 | 0.295
 
As shown above, fp32 model has completely same result as raw model in twice tests.
In tensorrt 6.0.1, the fp32 model has little less mAP than raw model, but
the model is converted through uff.