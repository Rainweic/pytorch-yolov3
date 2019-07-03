import cv2 as cv
import numpy as np
from .http_server import MainHandler
from torch.autograd import Variable
import torch
import tornado

def draw_bbox_label(image, classes, classIds, bboxes):
    for i in range(len(classIds)):
        classId = classIds[i]
        bbox = bboxes[i]
    
        cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))

        if classes:
            assert(classId < len(classes))
            label = '%s' % classes[classId]
        
        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = bbox[0]
        y = max(bbox[1], labelSize[1])
        cv.putText(image, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    return image

def process_video(args, net, classes):
    cap = cv.VideoCapture(args.video)

    # Get video information
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    save_cap = cv.VideoWriter("./output.mp4", cv.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (width,height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("Video is over")
            print("The video is output.mp4")
            break
        # deal with the frame
        classIds, bboxes = process_img(args, net, frame, classes)
        # draw bbox and text
        draw_bbox_label(frame, classes, classIds, bboxes)
        
        # Show frame
        cv.imshow("video", frame)
        cv.waitKey(1)

        # Save video 
        save_cap.write(frame)
    cap.release()
    save_cap.release()
    cv.destroyAllWindows()

def run_http(args, net, classes):
    settings = {'debug' : True}
    app = tornado.web.Application([
        (r'/rod', MainHandler, {
            "args": args,
            "net": net,
             "classes": classes}
        )
    ], **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    print("port: {}".format(args.port))
    http_server.bind(args.port)
    http_server.start(1)    # 0为多线程 1为单线程
    tornado.ioloop.IOLoop.instance().start()

def process_img(args, net, img, classes):
    """
    get bbox by process img
    """
    img_with_draw = img.copy()
    im_dim = img.shape[1], img.shape[0]
    im_dim = torch.FloatTensor(im_dim).repeat(1,2) 
    img = __prep_image(img, int(net.net_info["height"]))

    if torch.cuda.is_available():
        im_dim = im_dim.cuda()
        img = img.cuda()
    
    with torch.no_grad():
        output = net(Variable(img, volatile = True), torch.cuda.is_available())
    output = __write_results(output, args.confThreshold, num_classes=80, nms_conf = args.nmsThreshold)

    if type(output) == int:
        return [], []
        
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
    
    inp_dim = int(net.net_info["height"])
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    
    classIds = []
    bboxes = []
    # 生成bboxes
    for item in output:
        x1 = int(item[1].item())
        y1 = int(item[2].item())
        x2 = int(item[3].item())
        y2 = int(item[4].item())
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        bbox = [x1, y1, x2, y2, x_center, y_center]
        classId = int(item[-1])
        classIds.append(classId)
        bboxes.append(bbox)
    if args.image:
        image = draw_bbox_label(img_with_draw, classes, classIds, bboxes)
        cv.imwrite("output.jpg", image)
    return classIds, bboxes


def __letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv.resize(img, (new_w,new_h), interpolation = cv.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def __prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (__letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def __write_results(prediction, confidence, num_classes=80, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = __unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = __bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0

def __unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def __bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou
