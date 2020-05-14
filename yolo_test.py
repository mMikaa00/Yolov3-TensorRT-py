import argparse
from utils import detect_video, detect_image
from PIL import Image


# test
def detect_img(yolo):
    import cv2
    import numpy as np
    # while True:
    img = input('Input image filename:')
    if not img:
        return
    # dataset = open("dataset/2012_val.txt")
    # for line in dataset.readlines():
    #     img = line.split()[0]
    try:
        image = Image.open(img)
        img_origin = image.copy()
    except:
        print('Open Error! Try again!')
        exit()
    else:
        r_image = yolo.detect_image(image)
        img_output = np.asarray(r_image)
        img_output = np.concatenate([np.asarray(img_origin), np.asarray(r_image)], axis=1)
        cv2.imshow("test", cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR))
        if cv2.waitKey() & 0xFF == ord('q'):
            exit()


# FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file'
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions'
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions'
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use'
    )

    parser.add_argument(
        '--platform', type=str, required=False, default='tensorrt',
        help='Inference platform: tensorflow or tensorrt'
    )
    '''
    image detection mode
    '''
    parser.add_argument(
        "--image_path", type=str, required=False,
        help="Image input path"
    )

    parser.add_argument(
        "--image_output_path", default='', type=str, required=False,
        help="Image output path"
    )
    '''
    video detection mode
    '''

    parser.add_argument(
        "--video_path", type=str, required=False,
        help="Video input path"
    )

    parser.add_argument(
        "--video_output_path", default='', type=str, required=False,
        help="Video output path"
    )

    parser.add_argument(
        "--live", nargs='?', required=False,
        help="Live mode"
    )

    FLAGS = parser.parse_args()
    if FLAGS.platform == 'tensorrt':
        from yolo_tensorrt import YOLO
    else:
        from yolo_keras import YOLO

    if "image_path" in FLAGS:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        detect_image(YOLO(**vars(FLAGS)), FLAGS.image_path, FLAGS.image_output_path)
    elif "video_path" in FLAGS:
        print("local video mode")
        detect_video(YOLO(**vars(FLAGS)), FLAGS.video_path, FLAGS.video_output_path)
    elif "live" in FLAGS:
        print("live mode")
        detect_video(YOLO(**vars(FLAGS)), 0)
    else:
        print("Must specify at least video_input_path , image_input_path or live.  See usage with --help.")
