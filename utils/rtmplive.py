import subprocess as sp
import cv2 as cv
from . import utils

def live(args, net, classes):
    rtmpUrl = args.rtmpurl
    camera_path = args.video
    cap = cv.VideoCapture(camera_path)

    # Get video information
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # ffmpeg command
    command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'flv', 
            rtmpUrl]

    # 管道配置
    p = sp.Popen(command, stdin=sp.PIPE)
    
    # read webcamera
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("Opening camera is failed")
            break
        
        # process frame
        classIds, bboxes = utils.process_img(args, net, frame, classes)
        frame = utils.draw_bbox_label(frame, classes, classIds, bboxes)

        # write to pipe
        p.stdin.write(frame.tostring())
        p.stdin.flush()
