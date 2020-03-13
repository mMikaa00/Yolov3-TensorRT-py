import argparse
import sys
import os
import tensorflow as tf


def optimize_h5_model(model_path, output_path):
    # tf.enable_eager_execution()
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    model_path = os.path.expanduser(model_path)
    output_path = os.path.expanduser(output_path)
    model = tf.keras.models.load_model(model_path)
    name = os.path.basename(model_path)
    name = os.path.splitext(name)[0]
    temp_path = os.path.join('/tmp', name)
    print(temp_path)
    model.save(temp_path, save_format='tf')
    # tf.compat.v1.saved_model.save(model, temp_path)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=temp_path)
    converter.convert()
    converter.save(output_path)


def freeze_keras_model(model_path, output_path, keep_var_names=None):
    # First freeze the graph and remove training nodes.
    model_path = os.path.expanduser(model_path)
    output_path = os.path.expanduser(output_path)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}))  # CPU only
    tf.compat.v1.keras.backend.set_session(sess)
    tf.compat.v1.keras.backend.set_learning_phase(0)
    model = tf.compat.v1.keras.models.load_model(model_path)
    tf.compat.v1.keras.backend.set_learning_phase(0)
    if isinstance(model.input, list):
        input_names = [input.name for input in model.input]
    elif isinstance(model.input, tf.Tensor):
        input_names = [model.input.op.name]
    else:
        raise Exception('No input')
    output_names = [output.op.name for output in model.output]
    freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
    # print(freeze_var_names)
    print(input_names)
    print(output_names)

    tf.compat.v1.train.write_graph(sess.graph.as_graph_def(), '.', os.path.join(output_path, 'graph.pbtxt'),
                                   as_text=True)
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_names,
                                                                          freeze_var_names)
    frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
    # Save the model
    output_path = os.path.join(output_path, 'frozen.pb')
    with open(output_path, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())


def convert_keras2onnx(model_path, output_path):
    import keras2onnx
    import onnx
    # load keras model
    model = tf.compat.v1.keras.models.load_model(model_path)

    # convert to onnx model
    print(onnx.defs.onnx_opset_version())
    onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=10)
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    # # runtime prediction
    output_path = os.path.join(output_path, 'converted.onnx')
    onnx.save_model(onnx_model, output_path)


def main(args):
    if args.type == 'onnx':
        convert_keras2onnx(args.model_path, args.output_path)
    elif args.type == 'pb':
        freeze_keras_model(args.model_path, args.output_path)
    elif args.type == 'trt':
        optimize_h5_model(args.model_path, args.output_path)
    else:
        print("Type error")


def parse_arguments():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('--model_path', type=str,
                        help='Path of the model to be converted .', default='')
    parser.add_argument('--output_path', type=str,
                        help='Path of the model to be stored .', default='./optimized_model')
    parser.add_argument('--type', type=str,
                        help='Convert model to tyoe: onnx, pb or trt', default='pb')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
