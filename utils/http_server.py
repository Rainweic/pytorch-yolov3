import json
import base64
import time
import cv2
from tornado.options import define
import tornado.web
import numpy as np
from . import model
from . import utils


class MainHandler(tornado.web.RequestHandler):

    def initialize(self, args, net, classes):
        self.args = args
        self.net = net
        self.classes = classes

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with,content-type")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Access-Control-Max-Age", "3600");
   	
    #定义一个响应OPTIONS 请求，不用作任务处理
    def options(self):
        pass
    
    # get方法
    def get(self):
        self.write("运行正常")

    # post方法
    def post(self):
        '''
        request.body读取到的是二进制字符串 
        需要用decode("utf8")转为字符串
        再用json.loads()转为json对象
        '''  

        # 指定摄像头id
        post_id = self.get_argument("id")
        # 是否初始化该摄像头
        is_init = eval(self.get_argument("init"))
        # 是否关闭摄像头
        is_close = eval(self.get_argument("close"))
        # 从json中获取数据                 
        json_data_byte = self.request.body
        json_data_str = json_data_byte.decode('utf8')
        json_data_obj = json.loads(json_data_str)

        fps = int(json_data_obj.get("Fps"))
        sample = int(json_data_obj.get("SamplingRate"))
        width = int(json_data_obj.get("Width"))
        height = int(json_data_obj.get("Height"))
        image = self.base64_to_image(json_data_obj.get("ImageData"))
        points = json_data_obj.get("Poins")

        print(fps, sample, width, height)
        print(points)
        print("id", post_id, "is_close", is_close)
        print("is_init", is_init)

        if is_init is True:
            # 初始化摄像头查询
            pass

        elif is_init is False:
            if is_close is True:
                # 关闭摄像头查询
                pass
            elif is_close is False:
                t_start = time.time()
                # 处理图片
                classIds, bboxes, confidences = utils.process_img(self.args, self.net, image, self.classes)
                # TODO 做你要做的工作吧
                t_end = time.time()

                print("http：处理图片时间{}".format(t_end-t_start))
                print('writing...')
                print("result--------", results)

                self.write(json.dumps(results))

    def base64_to_image(self, base64_code):
        """
        函数说明:
        转换jpg图片的base64编码为opencv格式
        """
        lens = len(base64_code)
        lenx = lens - (lens % 4 if lens % 4 else 4)

        # try:
        #     img_data = base64.decodestring(strg[:lenx])
        # except:
        #     print("base64_to_image出错")

        # base64解码
        img_data = base64.b64decode(base64_code)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    
        return img


        
