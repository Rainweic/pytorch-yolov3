import argparse
import cv2 as cv
from utils import model
from utils import utils
import time

def getargs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weight",
        default = "./yolo_v3_config/yolov3.weights",
        type = str,
        help = "The path of yolov3.weights"
    )
    parser.add_argument(
        "--coco_names",
        default = './yolo_v3_config/coco.names',
        type = str,
        help = "The path of coco.names"
    )
    parser.add_argument(
        "--yolov3_cfg",
        default = "./yolo_v3_config/yolov3.cfg",
        type = str,
        help = "The path of yolov3.cfg"
    )
    parser.add_argument(
        "--confThreshold",
        default = 0.5,
        type = float,
        help = "Confidence threshold"
    )
    parser.add_argument(
        "--nmsThreshold",
        default = 0.4,
        type = float,
        help = "Non-maximum suppression threshold"
    )
    parser.add_argument(
        "--inpWidth",
        default = 416,
        type = int,
        help = "Width of network's input image"
    )
    parser.add_argument(
        "--inpHeight",
        default = 416,
        type = int,
        help = "Height of network's input image"
    )
    parser.add_argument(
        "--image",
        default = None,
        type = str,
        help = "path of image"
    )
    parser.add_argument(
        "--video",
        default = None,
        type = str,
        help = "path of video"
    )
    parser.add_argument(
        "--run_http",
        default = False,
        type = bool,
        help = "Open http server"
    )
    parser.add_argument(
        "--port",
        default = 8080,
        type = int,
        help = "http server's port"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # load model
    args = getargs()
    net, classes = model.load_model(args)

    if args.image:
        image = cv.imread(args.image)
        utils.process_img(args, net, image, classes)
    elif args.video:
        utils.process_video(args, net, classes)
    elif args.run_http:
        utils.run_http(args, net, classes)