from ultralytics import YOLO
import torch
import cv2 as cv
import numpy as np
import time
import argparse


class Detector(object):

    def __init__(self, weight_path):
        self.weight_path = weight_path
        # 载入模型
        self.model = YOLO(self.weight_path)
        # get class labels
        self.objs_labels = self.model.names  
        # set confidence
        self.conf = 0.5
        # 获取视频流
        # self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        # 检测视频
        self.cap = cv.VideoCapture("test_videos/1713429986.avi")

        
    def detect(self):
        while True:
            # ret -> ture or false 有没有读取到画面
            # frame 读取的每一帧画面
            ret, frame = self.cap.read()
            # 开始时间
            # start_time = time.time()
            if ret:
                # 画面翻转
                frame = cv.flip(frame, 1)
                # 转为RGB格式
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                # 检测推理
                result = list(self.model(frame, conf=self.conf, stream=True))[0]
                boxes = result.boxes.cpu().numpy()  # Boxes object for bbox outputs, convert to numpy array
                keypoints = result.keypoints.cpu().numpy()
                # print(boxes)
                # print(keypoints)
                # 绘制边界框
                for box in boxes.data:
                    l,t,r,b = box[:4].astype(np.int32) # left, top, right, bottom
                    conf, id = box[4:] # confidence, class
                    # 绘制框
                    cv.rectangle(frame, (l,t), (r,b), (0,0,255), 2)
                    # 绘制类别+置信度（格式：98.1%）
                    cv.putText(frame, f"{self.objs_labels[id]} {conf*100:.1f}%", (l, t-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                for keypoints in keypoints.data:
                    for idx, point in enumerate(keypoints):
                        x, y = point[:2].astype(np.int32)
                        cv.circle(frame, (x, y), 5, (0, 255, 255), -1)
                        cv.putText(frame, str(idx), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


                # 结束时间
                end_time = time.time()
                # FPS
                # fps = 1 / (end_time - start_time)
                # 绘制FPS
                # cv.putText(frame, "FPS: " + str(int(fps)), (10, 40), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                # 显示画面
                cv.imshow("frame", frame)
                # 条件判断
                key = cv.waitKey(2000)
                if key & 0xFF == ord('q'):
                    break
                if key == 32:
                    cv.waitKey(0)
                    continue
            else:
                break
        # 释放
        self.cap.release()
        cv.destroyAllWindows()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="weights/yolov8s-pose.pt")
    args = parser.parse_args()
    detector = Detector(args.weight)
    detector.detect()