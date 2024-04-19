import cv2 as cv
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from sam_model import SegmentModel
import argparse

class SamCutouts(object):

    def __init__(self, img_path):
        # 读取图片
        self.img_path = img_path
        self.img = cv.imread(self.img_path)
        self.img_copy = self.img.copy()
        # 加载模型
        self.model_path = "./weights/sam_vit_h_4b8939.pth"
        self.model = SegmentModel(model_path=self.model_path, model_type="vit_h")
        # 模型初始化
        self.model.init_embedding(self.img_path)
        # 前景点
        self.segment_pts = []
        # 背景点
        self.ignore_pts = []

    # 设置鼠标事件
    def setMouseEvent(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONUP:     # 点击左键
            self.img_copy = self.img.copy()
            print(f"点击左键，坐标为({x}，{y})")
            self.segment_pts.append((x, y))
            # 更新抠图
            self.cutouts()
        elif event == cv.EVENT_RBUTTONUP:     # 点击右键
            self.img_copy = self.img.copy()
            print(f"点击右键，坐标为({x}，{y})")
            self.ignore_pts.append((x, y))
            # 更新抠图
            self.cutouts()

    # mask抠图
    def cutouts(self):
        # 输入点以及标签
        input_points = np.array(self.segment_pts + self.ignore_pts)
        input_lables = np.array([1] * len(self.segment_pts) + [0] * len(self.ignore_pts))
        # 预测推理
        masks, scores = self.model.predict(input_points, input_lables)
        # 是否选择得分最高的结果
        return_by_maxscore = True
        if return_by_maxscore:
            idx = np.argmax(scores)     # 获取得分最高的索引
            mask = masks[idx]
        else:
            mask = masks[0]
        mask = mask.astype(np.uint8) * 255
        self.img_copy[mask == 0] = 255
        

    # main 函数
    def main(self):
        cv.namedWindow("img", cv.WINDOW_KEEPRATIO)
        cv.setMouseCallback("img", self.setMouseEvent)      # 设置鼠标事件
        while True:
            cv.imshow("img", self.img_copy)
            key = cv.waitKey(20)
            if key == 27 :
                print("end cutouts!")
                break
            elif key & 0xFF == ord("r"):
                self.img_copy = self.img.copy()
                self.segment_pts = []
                self.ignore_pts = []
                print("重置完毕！")
            elif key & 0xFF == ord("w"):
                cv.imwrite("./test_imgs/cutouts.jpg", self.img_copy)
                print("保持图片！")
        cv.destroyAllWindows()



if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_file", type=str, default="./test_imgs/great_cat.jpg")
    args = parser.parse_args()
    cutouts = SamCutouts(args.img_file)
    cutouts.main()