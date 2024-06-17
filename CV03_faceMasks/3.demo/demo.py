import torch
import numpy as np
import os
import glob
import cv2 as cv
import tqdm
import time
from PIL import Image,ImageDraw,ImageFont
from model.SSD import FaceMaskDetection
from model.FACENET import InceptionResnetV1



class FaceMaskDetector(object):

    def __init__(self):
        # 加载人脸检测模型
        self.face_detector = FaceMaskDetection('./weights/face_mask_detection.pb', margin=0, GPU_ratio=0.1)
        # device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 加载人脸识别模型
        self.mask_detector = InceptionResnetV1(is_train=False, embedding_length=128, num_classes=14575).to(device=self.device)
        # 加载权重
        self.mask_detector.load_state_dict(torch.load('./weights/facenet_best.pt',map_location=self.device))
        # 加载佩戴和未佩戴标志
        self.mask_class_overlay = self.getMaskClassPngs()
        # 加载目标人的特征
        # name_list支持：恩培,恩培_1,,恩培_2形式，与known_embedding对应
        self.name_list,self.known_embedding = self.getFaceFeats()
        # 增加一个未知人员
        self.name_list.append('未知')
        # 生成每个人的名称PNG图片（以解决中文显示问题）
        self.name_png_list = self.getNamePngs(self.name_list)

    # 加载佩戴和未佩戴标志
    def getMaskClassPngs(self):
        labels = ['masked','without_mask']
        overlay_list = []
        for label in labels:
            fileName = './images/%s.png' % (label)
            overlay = cv.imread(fileName,cv.COLOR_RGB2BGR)
            overlay = cv.resize(overlay,(0,0), fx=0.2, fy=0.2)
            overlay_list.append(overlay)
        return overlay_list

    # 读取文件
    def readPngFile(self,fileName):
        # 解决中文路径问题
        png_img = cv.imdecode(np.fromfile(fileName,dtype=np.uint8),-1)
        # 转为BGR，变成3通道
        png_img = cv.cvtColor(png_img,cv.COLOR_RGB2BGR)
        png_img = cv.resize(png_img,(0,0), fx=0.4, fy=0.4)
        return png_img

    # 得到裁剪后的人脸照片
    def getCropedFace(self, img_path, conf_thresh=0.5):
        # 读取图片
        # 解决中文路径问题
        img_ori = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img_ori is None:
            return None
        # 转RGB
        img = cv.cvtColor(img_ori,cv.COLOR_BGR2RGB)
        # 缩放
        img = cv.resize(img,self.face_detector.img_size)
        # 转float32
        img = img.astype(np.float32)
        # 归一
        img /= 255
        # 增加维度
        img_4d = np.expand_dims(img,axis=0)
        # 原始高度和宽度
        ori_h,ori_w = img_ori.shape[:2]
        bboxes, re_confidence, re_classes, re_mask_id = self.face_detector.inference(img_4d,ori_h,ori_w)
        for index,bbox in enumerate(bboxes):
            class_id = re_mask_id[index] 
            l,t,r,b = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

            croped_face = img_ori[t:b,l:r]
            return croped_face 
        # 都不满足
        return None

    # 图片预处理
    def imgPreprocess(self,img):
        # 转为float32
        img = img.astype(np.float32)
        # 缩放
        img = cv.resize(img,(112,112))
        # BGR 2 RGB
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        # h,w,c 2 c,h,w
        img = img.transpose((2,0,1))
        # 归一化[0,255] 转 [-1,1]
        img = (img - 127.5) / 127.5
        # 增加维度
        # img = np.expand_dims(img,0)
        return img

    # 得到特征向量库
    def getFaceFeats(self):
        # 记录名字
        name_list = []
        # 输入网络的所有人脸图片
        known_faces_input = []
        # 遍历
        known_face_list = glob.glob('./images/origin/*')
        for face in tqdm.tqdm(known_face_list,desc='处理目标人脸...'):
            name = face.split('\\')[-1].split('.')[0]
            name_list.append(name)
            # 裁剪人脸
            croped_face = self.getCropedFace(face)
            if croped_face is None:
                print('图片：{} 未检测到人脸，跳过'.format(face))
                continue
            # 预处理
            img_input = self.imgPreprocess(croped_face)
            known_faces_input.append(img_input)
        # 转为np
        faces_input = np.array(known_faces_input)
        # 转tensor并放到GPU
        tensor_input = torch.from_numpy(faces_input).to(self.device)
        # 得到所有的embedding,转numpy
        known_embedding = self.mask_detector(tensor_input).detach().cpu().numpy()
        return name_list,known_embedding

    # 生成每个人的名称PNG图片（以解决中文显示问题）
    def getNamePngs(self, name_list):
        # 先将['恩培','恩培_1','恩培_2','小明','小明_1','小明_2']变成['恩培','小明']
        real_name_list = []
        for name in name_list:
            real_name = name.split('_')[0]
            if real_name not in real_name_list:
                real_name_list.append(real_name)
        pngs_list = {}
        for name in tqdm.tqdm(real_name_list,desc='生成人脸标签PNG...'):
            filename = './images/name_png/'+name+'.png'
            # 如果存在，直接读取
            if os.path.exists(filename):
                png_img = self.readPngFile(filename)
                pngs_list[name] = png_img
                continue
            # 如果不存在，先生成
            # 背景
            bg = Image.new("RGBA",(400,100),(0,0,0,0))
            # 添加文字
            d = ImageDraw.Draw(bg)
            font  = ImageFont.truetype('./fonts/MSYH.ttc',80,encoding="utf-8")
            if name == '未知':
                color = (0,0,255,255)
            else:
                color = (0,255,0,255)
            d.text((0,0),name,font=font,fill=color)
            # 保存
            bg.save(filename)
            # 再次检查
            if os.path.exists(filename):
                png_img = self.readPngFile(filename)
                pngs_list[name] = png_img
        return pngs_list


    def main(self):
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv.CAP_PROP_FPS)
        #创建新视频
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        output = './results/result.mp4'
        writer = cv.VideoWriter(output, fourcc, fps, (WIDTH, HEIGHT))

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            # 水平翻转
            frmae = cv.flip(frame, 1)
            # 变换颜色通道
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # 检测人脸
            # 缩放成ssd需要size
            img = cv.resize(img, self.face_detector.img_size)
            # 转为float32
            img = img.astype(np.float32)
            # 归一
            img /= 255
            # 增加维度
            img_4d = np.expand_dims(img, axis=0)
            # 推理
            bboxes, re_confidence, re_classes, re_mask_id = self.face_detector.inference(img_4d,HEIGHT,WIDTH)
            # 检测每一个人脸
            for index,bbox in enumerate(bboxes):
                class_id = re_mask_id[index] 
                conf  = re_confidence[index]
                if class_id == 0:
                    color = (0, 255, 0)  # 戴口罩
                elif class_id == 1:
                    color = (0, 0, 255)  # 没带口罩
                l,t,r,b = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                # 裁剪人脸
                crop_face = frame[t:b, l:r]
                # 人脸识别（to 128维特征向量）
                # 转为float32
                img = crop_face.astype(np.float32)
                # 缩放
                img = cv.resize(img, (112, 112))
                # BGR2RGB
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                # h,w,c 2 c,h,w
                img = img.transpose((2, 0, 1))
                # 归一化[0. 255] to [-1. 1]
                img = (img - 127.5) / 127.5
                # 扩展维度
                img_input = np.expand_dims(img, 0)
                # print(img_input.shape)
                # 转为tensor放在GPU上
                tensor_input = torch.from_numpy(img_input).to(self.device)
                # 得到embedding
                embedding  = self.mask_detector(tensor_input)
                embedding = embedding.detach().cpu().numpy()
                # 计算距离
                dist_list = np.linalg.norm((embedding-self.known_embedding), axis=1)
                # 最小距离索引
                min_index = np.argmin(dist_list)
                # 识别人民和距离
                pred_name = self.name_list[min_index]
                # 最短距离
                min_dist = dist_list[min_index]
                if min_dist < 1:
                    # 识别到人
                    # 人名png
                    real_name = pred_name.split('_')[0]
                    name_overlay = self.name_png_list[real_name]
                else:
                    # 未识别到，加载未知
                    name_overlay = self.name_png_list['未知']
                # √和×标志
                class_overlay = self.mask_class_overlay[class_id]
                # 拼接两个PNG
                overlay = np.zeros((40, 210, 3), np.uint8)
                overlay[:40, :40] = class_overlay
                overlay[:40, 50:210] = name_overlay
                # 覆盖显示
                overlay_h, overlay_w = overlay.shape[:2]
                # 覆盖范围
                overlay_l = l
                overlay_t = t - overlay_h - 20
                overlay_r = l + overlay_w
                overlay_b = overlay_t + overlay_h
                # 判断边界
                if overlay_t > 0 and overlay_r < WIDTH:
                    overlay_copy=cv.addWeighted(frame[overlay_t:overlay_b, overlay_l:overlay_r ],1,overlay,20,0)
                    frame[overlay_t:overlay_b, overlay_l:overlay_r ] =  overlay_copy
                print(pred_name,min_dist)

                cv.rectangle(frame,(l,t),(r,b),color,2)
                fps = 1/ (time.time()- start_time)
                cv.putText(frame,str(round(fps,2)),(50,50),cv.FONT_ITALIC,1,(0,255,0),2)

            # 显示画面
            cv.imshow("demo", frame)
            # 保存画面
            writer.write(frame)
            # 条件退出
            if cv.waitKey(1) & 0xff == ord('q'):
                break

        # 释放
        writer.release()
        cap.release()
        cv.destroyAllWindows()



fmd = FaceMaskDetector()
fmd.main()