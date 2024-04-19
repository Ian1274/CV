import numpy as np
import cv2 as cv


def getIuo(box_A, box_B):

    sum_A = (box_A[2] - box_A[0]) * (box_A[3] - box_A[1])
    sum_B = (box_B[2] - box_B[0]) * (box_B[3] - box_B[1])
    
    iou_l = max(box_A[0], box_B[0])
    iou_t = max(box_A[1], box_B[1])
    iou_r = min(box_A[2], box_B[2])
    iou_b = min(box_A[3], box_B[3])

    if iou_t >= iou_b or iou_l >= iou_r:
        return 0, 0
    else:
        sum_intersection = (iou_r - iou_l) * (iou_b - iou_t)
        iou = sum_intersection / (sum_A + sum_B - sum_intersection)
        return [iou_l, iou_t, iou_r, iou_b], iou




if __name__ == "__main__":
    box_A = [100, 100, 400, 400]
    box_B = [50, 50, 300, 300]
    black = np.zeros((440, 660, 3), np.int8)
    cv.rectangle(black, box_A[:2], box_A[2:], (0, 255, 0), 1)
    cv.rectangle(black, box_B[:2], box_B[2:], (0, 255, 255), 1)

    iou_l = max(box_A[0], box_B[0])
    iou_t = max(box_A[1], box_B[1])
    iou_r = min(box_A[2], box_B[2])
    iou_b = min(box_A[3], box_B[3])

    cv.circle(black, (iou_l, iou_t), 10, (255, 0, 0), -1)
    cv.circle(black, (iou_r, iou_b), 10, (255, 0, 0), -1)

    iou_box, iou = getIuo(box_A, box_B)

    # print(type(iou_box))
    # print(iou)

    cv.rectangle(black, iou_box[:2], iou_box[2:], (255, 255, 0), -1)
    cv.imshow("iou", black)
    cv.waitKey(0)
    cv.destroyAllWindows()

