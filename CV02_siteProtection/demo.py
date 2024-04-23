from ultralytics import YOLO
import torch 
import cv2 as cv
import numpy as np
import pandas as pd
import time
import math
import glob
import os
import argparse

class Detector(object):

    def __init__(self, weight_path, weight_pose_path, video_path):
        self.weight_path = weight_path
        self.weight_pose_path = weight_pose_path
        self.video_path = video_path
        # 加载模型
        self.model = YOLO(self.weight_path)
        self.model_pose = YOLO(self.weight_pose_path)
        self.confidence = 0.7
        self.label_names = self.model.names     # {0: 'person', 1: 'vest', 2: 'blue helmet', 3: 'red helmet', 4: 'white helmet', 5: 'yellow helmet'}
        self.label_names_swap = {v : k for k, v in self.label_names.items()}
        # 视频流
        self.cap = cv.VideoCapture(self.video_path)
        self.frame_w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # 视频编码器
        self.fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        # 帧数
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        # 写入视频
        self.out = cv.VideoWriter("./results/result.mp4", self.fourcc, self.fps, (self.frame_w, self.frame_h))
        # 贴纸路径
        self.icons_path = "./icons"
        # 贴纸图标字典 {"name":cv_img,......}
        self.icons_dic = {}
        # 载入贴纸图标
        self.get_icons(self.icons_path)
        # 颜色 Blue Green Red
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # 获取贴图图标
    def get_icons(self, icons_path):
        icon_paths_list = glob.glob(icons_path + "/*.png")
        for icon_path in icon_paths_list:
            icon_name = icon_path.split(os.sep)[-1].split('.png')[0]
            icon_img = cv.imread(icon_path)
            icon_img = cv.resize(icon_img, (50, 50))
            self.icons_dic[icon_name] = icon_img

    # 获取关键点列表0-4个关键点中第一个存在的关键点
    def getNo0keypoint(self, keypoints, index):
        if index <=4 :
            if keypoints[index][0] == 0:
                return self.getNo0keypoint(keypoints, index + 1)
            else:
                return keypoints[index]
        else:
            print("没有检测到头部关键点（0-4）")
            return 0

    # 增加贴图
    def add_icon(self, src, icon, roipots):
        x, y = roipots
        # print(x, y)
        cut_h, cut_w = icon.shape[:2]
        # print(cut_h, cut_w)
        roi = src[y:y+cut_h, x:x+cut_w]
        icon_gray = cv.cvtColor(icon, cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(icon_gray, 0, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        cut_2mask = cv.bitwise_and(roi, roi, mask=mask_inv)
        icon_2mask = cv.bitwise_and(icon, icon, mask=mask)
        dst = cv.add(cut_2mask, icon_2mask)
        src[y:y+cut_h, x:x+cut_w] = dst
        # return src

    # 获取两个框的IOU值
    def getIou(self, box_A, box_B):
        sum_A = (box_A[2] - box_A[0]) * (box_A[3] - box_A[1])
        sum_B = (box_B[2] - box_B[0]) * (box_B[3] - box_B[1])
        iou_l = max(box_A[0], box_B[0])
        iou_t = max(box_A[1], box_B[1])
        iou_r = min(box_A[2], box_B[2])
        iou_b = min(box_A[3], box_B[3])
        if iou_t >= iou_b or iou_l >= iou_r:
            return 0
        else:
            sum_intersection = (iou_r - iou_l) * (iou_b - iou_t)
            iou = sum_intersection / (sum_A + sum_B - sum_intersection)
            return iou

    # 包含人员信息[{"index":index, "person_info":[人物框坐标], "keypoints_info":[17个关键点坐标], "vest_info":[安全服坐标], "helmet_info":[安全帽坐标, 对应id]}， {...}, ...]
    def get_person_info_list(self, person_lists, keypoints_lists, vest_lists, helmet_lists, frame):
        person_info_list = []
        if len(person_lists) != 0:
            for index, person in enumerate(person_lists):
                person_l, person_t, person_r, person_b = person
                person_info_dic = {}
                # 添加序号
                person_info_dic["index"] = index
                # 添加人物框坐标
                person_info_dic["person_info"] = person
                # 关键点、衣服、帽子初始信息设置为None
                person_info_dic["keypoints_info"] = None
                person_info_dic["vest_info"] = None
                person_info_dic["helmet_info"] = None
                # 匹配与人物相符的关键点s坐标
                for index, keypoints in enumerate(keypoints_lists):
                    keypoint_head_x,  keypoint_head_y = self.getNo0keypoint(keypoints, 0)[:2]
                    keypoint_body_05_x,  keypoint_body_05_y = keypoints[5][:2]
                    keypoint_body_06_x,  keypoint_body_06_y = keypoints[6][:2]
                    keypoint_body_x,  keypoint_body_y = (keypoint_body_05_x + keypoint_body_06_x)/2, (keypoint_body_05_y + keypoint_body_06_y)/2
                    if person_l < keypoint_head_x < person_r and person_t < keypoint_head_y < person_b:
                        # 将此关键点s加入到person_info_dict中
                        person_info_dic["keypoints_info"] = keypoints
                        # 匹配与人物相符的安全服坐标
                        if len(vest_lists) != 0:
                            for vest in vest_lists:
                                vest_l, vest_t, vest_r, vest_b = vest
                                vest_w, vest_h = vest_r - vest_l, vest_b - vest_t
                                keypoint_body_box = [keypoint_body_x - vest_w/2, keypoint_body_y - vest_h/2, keypoint_body_x + vest_w/2, keypoint_body_y + vest_h/2]
                                iou_body_vest = self.getIou(keypoint_body_box, vest)
                                # print(type(iou_body_vest), iou_body_vest)
                                if iou_body_vest > 0.2:
                                    # 将此vest坐标加入到person_info_dict中
                                    person_info_dic["vest_info"] = vest
                                    break
                        else:
                            person_info_dic["vest_info"] = None
                        # 匹配与人物相符的安全帽坐标
                        if len(helmet_lists) != 0:
                            for helmet in helmet_lists:
                                helmet_l, helmet_t, helmet_r, helmet_b, helmet_id = helmet
                                helmet_diagonal = math.sqrt((helmet_r - helmet_l)**2 + (helmet_b - helmet_t)**2)
                                keypoint_head_box = [keypoint_head_x - helmet_diagonal/2, keypoint_head_y - helmet_diagonal/2, keypoint_head_x + helmet_diagonal/2, keypoint_head_y + helmet_diagonal/2]
                                iou_head_helmet = self.getIou(keypoint_head_box, helmet[:4])
                                
                                # cv.rectangle(frame, (int(keypoint_head_x - helmet_diagonal/2), int(keypoint_head_y - helmet_diagonal/2)), (int(keypoint_head_x + helmet_diagonal/2), int(keypoint_head_y + helmet_diagonal/2)), (0, 255, 0), 1)
                                # cv.rectangle(frame, (helmet_l, helmet_t), (helmet_r, helmet_b), (0, 255, 0), 1)

                                # print(type(iou_head_helmet), iou_head_helmet)
                                if iou_head_helmet > 0:
                                    # 将此helmet坐标加入到person_info_dict中
                                    person_info_dic["helmet_info"] = helmet
                                    break
                        else:
                            person_info_dic["helmet_info"] = None
                person_info_list.append(person_info_dic)
        else:
            print("没有检测到人物")
        return person_info_list

        
    # 渲染画面
    def render(self, frame, person_infos_lists):
        for person_info_dic in person_infos_lists:  # {"index":index, "person_info":[人物框坐标], "keypoints_info":[17个关键点坐标], "vest_info":[安全服坐标], "helmet_info":[安全帽坐标, 对应id]}
            person_l, person_t, person_r, person_b = person_info_dic["person_info"]
            cv.rectangle(frame, (person_l, person_t), (person_r, person_b), self.colors[0], 2)
            if person_t-60 >= 0:
                self.add_icon(frame, self.icons_dic["person"], (person_l, person_t-60))
                # 安全服渲染
                if person_info_dic["vest_info"] is not None:
                    vest_l, vest_t, vest_r, vest_b = person_info_dic["vest_info"]
                    cv.rectangle(frame, (vest_l, vest_t), (vest_r, vest_b), self.colors[1], 2)
                    self.add_icon(frame, self.icons_dic["vest_on"], (person_l+60, person_t-60))
                else:
                    self.add_icon(frame, self.icons_dic["vest_off"], (person_l+60, person_t-60))
                # 安全帽渲染
                if person_info_dic["helmet_info"] is not None:
                    helmet_l, helmet_t, helmet_r, helmet_b, helmet_id = person_info_dic["helmet_info"]
                    cv.rectangle(frame, (helmet_l, helmet_t), (helmet_r, helmet_b), self.colors[2], 2)
                    if helmet_id == 2:
                        self.add_icon(frame, self.icons_dic["hat_bule"], (person_l+120, person_t-60))
                    elif helmet_id == 3:
                        self.add_icon(frame, self.icons_dic["hat_red"], (person_l+120, person_t-60))
                    elif helmet_id == 4:
                        self.add_icon(frame, self.icons_dic["hat_white"], (person_l+120, person_t-60))
                    elif helmet_id == 5:
                        self.add_icon(frame, self.icons_dic["hat_yellow"], (person_l+120, person_t-60))
                else:
                    self.add_icon(frame, self.icons_dic["hat_off"], (person_l+120, person_t-60))
            else:
                self.add_icon(frame, self.icons_dic["person"], (person_l, 0))
                # 安全服渲染
                if person_info_dic["vest_info"] is not None:
                    vest_l, vest_t, vest_r, vest_b = person_info_dic["vest_info"]
                    cv.rectangle(frame, (vest_l, vest_t), (vest_r, vest_b), self.colors[1], 2)
                    self.add_icon(frame, self.icons_dic["vest_on"], (person_l+60, 0))
                else:
                    self.add_icon(frame, self.icons_dic["vest_off"], (person_l+60, 0))
                # 安全帽渲染
                if person_info_dic["helmet_info"] is not None:
                    helmet_l, helmet_t, helmet_r, helmet_b, helmet_id = person_info_dic["helmet_info"]
                    cv.rectangle(frame, (helmet_l, helmet_t), (helmet_r, helmet_b), self.colors[2], 2)
                    if helmet_id == 2:
                        self.add_icon(frame, self.icons_dic["hat_blue"], (person_l+120, 0))
                    elif helmet_id == 3:
                        self.add_icon(frame, self.icons_dic["hat_red"], (person_l+120, 0))
                    elif helmet_id == 4:
                        self.add_icon(frame, self.icons_dic["hat_white"], (person_l+120, 0))
                    elif helmet_id == 5:
                        self.add_icon(frame, self.icons_dic["hat_yellow"], (person_l+120, 0))
                else:
                    self.add_icon(frame, self.icons_dic["hat_off"], (person_l+120, 0))

    def main(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                # 目标检测
                results = list(self.model(frame, conf=self.confidence, stream=True))[0]
                boxes = results.boxes.cpu().numpy()  # Boxes object for bbox outputs, convert to numpy array
                # 关键点检测
                results_pose = list(self.model_pose(frame, conf=self.confidence, stream=True))[0]
                keypoints = results_pose.keypoints.cpu().numpy()
                # 人员, 衣服, 安全帽信息
                person_lists = []
                vest_lists = []
                helmet_lists = []
                for box in boxes.data:
                    l,t,r,b = box[:4].astype(np.int32) # left, top, right, bottom
                    conf, id = box[4:] # confidence, class
                    if id == 0:
                        person_lists.append([l, t, r, b])
                    elif id == 1:
                        vest_lists.append([l, t, r, b])
                    else:
                        helmet_lists.append([l, t, r, b, id])
                # 获取人员信息
                person_infos_lists = self.get_person_info_list(person_lists, keypoints.data, vest_lists, helmet_lists, frame)
                # 渲染画面
                self.render(frame, person_infos_lists)
                # 写入帧
                self.out.write(frame)  
                # 显示画面
                cv.imshow("demo", frame)
                # 判断退出
                key = cv.waitKey(1000)
                if key & 0xFF == ord('q'):
                    break
                elif key == 32:
                    cv.waitKey(0)
                    continue
            else:
                print("未检测到画面，break!")
                break
        self.cap.release()
        self.out.release()
        cv.destroyAllWindows()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="siteProtection/64084/weights/best.pt")
    parser.add_argument("--weight_pose", type=str, default="weights/yolov8s-pose.pt")
    parser.add_argument("--video", type=str, default="test_videos/1713429986.avi")
    args = parser.parse_args()
    detector = Detector(args.weight, args.weight_pose, args.video)
    detector.main()