# 测试
# 图片输入网络得到embedding A
# 计算 A（戴口罩）与数据库中其他人脸的L2距离
# 找到距离最近的人脸

# 图片预处理（按照训练流程）
# 0.转为float32
# 1.缩放 112 x 112
# 2.转为RGB
# 3.通道顺序由h,w,c 转为c,h,w
# 4.归一化到[-1,1]
# 5.增加维度

# 视频中检测
# FAISS测试

# 加载模型
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from model.FACENET import InceptionResnetV1
from torchsummary import summary

class Demo(object):
    
    '''
    1.输入一张测试图片，和一组用来比较的库；
    2.对所有的图片都要进行人脸裁剪，以及图片预处理；
    3.比较测试图片与库的embedding；
    '''

    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 实例化facenet
        self.facenet = InceptionResnetV1(is_train=False,embedding_length=128,num_classes=14575).to(self.device)
        # 从训练文件中加载
        self.facenet.load_state_dict(torch.load(self.model_path))
        # 加载检测模型
        self.face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt','./weights/res10_300x300_ssd_iter_140000.caffemodel')

    # 定义函数：裁剪人脸
    def getCropedFace(self, img_file, conf_thresh=0.5 ):
        """
        将图片进行人脸裁剪
        @param:
            img_file: str 文件名
            conf_thresh: float 置信度预支
            w_thresh,h_thresh: 人脸长度宽度阈值，小于它则丢弃
        @return
            croped_face: numpy img 裁剪后的人脸，如果没有符合条件的，则返回None
        """
        # 读取图片
        # img = cv2.imread(img_file)
        # 解决中文路径问题
        img=cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
        if img is None:
            return None
        # 画面原来高度和宽度
        img_height,img_width = img.shape[:2]
        # 缩放图片
        img_resize = cv2.resize(img,(300,300))
        # 图像转为blob
        img_blob = cv2.dnn.blobFromImage(img_resize,1.0,(300,300),(104.0, 177.0, 123.0))
        # 输入
        self.face_detector.setInput(img_blob)
        # 推理
        detections = self.face_detector.forward()
        # 查看检测人脸数量
        num_of_detections = detections.shape[2]
        # 遍历人脸
        for index in range(num_of_detections):
            # 置信度
            detection_confidence = detections[0,0,index,2]
            # 挑选置信度，找到一个人返回
            if detection_confidence > conf_thresh:
                # 位置
                locations = detections[0,0,index,3:7] * np.array([img_width,img_height,img_width,img_height])
                # 矩形坐标
                l,t,r,b  = locations.astype('int')
                # 长度宽度判断
                w = r - l
                h = b - t
                # print(w,h)
                croped_face = img[t:b,l:r]
                return croped_face
        # 都不满足
        return None

    # 定义函数：图片预处理
    def imgPreprocess(self, img):
        # 转为float32
        img = img.astype(np.float32)
        # 缩放
        img = cv2.resize(img,(112,112))
        # BGR 2 RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # h,w,c 2 c,h,w
        img = img.transpose((2,0,1))
        # 归一化[0,255] 转 [-1,1]
        img = (img - 127.5) / 127.5
        # 增加维度
        # img = np.expand_dims(img,0)
        return img

    # 主函数
    def main(self, path_testImg, dir_imgs):

        test_croped_face = self.getCropedFace(path_testImg)
        # 预处理
        test_img_input = self.imgPreprocess(test_croped_face)
        # 输入网络
        # 扩展维度
        test_img_input = np.expand_dims(test_img_input,0)
        # 转tensor并放到GPU
        tensor_test_input = torch.from_numpy(test_img_input).to(self.device)
        # 得到embedding
        test_embedding = self.facenet(tensor_test_input)
        # 转numpy
        test_embedding = test_embedding.detach().cpu().numpy()

        # 将所有对比库图片输入网络
        known_face_list = glob.glob(dir_imgs + '/*.jpg')
        # 记录名字
        name_list = []
        # 输入网络的所有人脸图片
        known_faces_input = []
        # 遍历
        for face in known_face_list:
            name = face.split('\\')[-1].split('.')[0]
            name_list.append(name)
            croped_face = self.getCropedFace(face)
            img_input = self.imgPreprocess(croped_face)
            known_faces_input.append(img_input)
        # 转为Nummpy
        faces_input = np.array(known_faces_input)
        # 转tensor并放到GPU
        tensor_input = torch.from_numpy(faces_input).to(self.device)
        # 得到所有的embedding
        known_embedding = self.facenet(tensor_input)
        # 转numpy
        known_embedding = known_embedding.detach().cpu().numpy()

        # 计算测试图片与参考图片embedding之间距离
        dist_list = np.linalg.norm((test_embedding-known_embedding), axis=1)
        # 最小距离索引
        min_index = np.argmin(dist_list)
        # 识别人名与距离
        print(name_list[min_index])
        print(dist_list[min_index])


test_demo = Demo('./save_model/facenet_best.pt')
test_demo.main('./images/test/unknown_2.jpg', './images/test/db')





























# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实例化
facenet = InceptionResnetV1(is_train=False,embedding_length=128,num_classes=14575).to(device)

# 从训练文件中加载
facenet.load_state_dict(torch.load('./save_model/facenet_best.pt'))

# print(facenet.eval())

# 加载检测模型
face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt','./weights/res10_300x300_ssd_iter_140000.caffemodel')


