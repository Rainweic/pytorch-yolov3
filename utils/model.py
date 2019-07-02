import cv2 as cv
from .darknet import Darknet
import torch

def load_model(args):
    '''
    load model
    '''
    # Load names of classes
    print("loading coco_names...")
    classesFile_path = args.coco_names
    classes = load_classesFile(classesFile_path)

    # Load config file and weight
    print("loading yolov3 config file & weight...")
    modelConfiguration = args.yolov3_cfg;
    modelWeights = args.weight;
    
    net = Darknet(modelConfiguration)
    net.load_weights(modelWeights)
    net.net_info["height"] = args.inpHeight


    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    print("model is loaded...")
    return net, classes

def load_classesFile(path):
    '''
    load classes file
    '''
    classes = None
    with open(path, "rt") as f:
        classes = f.read().rstrip('\n').split('\n')
        classes = [ item for item in classes if item[0] != "#"]
    return classes

