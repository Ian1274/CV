import cv2 as cv
import imageio

def readVideo(videoPath):
    video_cap = cv.VideoCapture(videoPath)
    frame_count = 0
    all_frames = []
    while True:
        ret, frame = video_cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
            all_frames.append(frame)
            frame_count = frame_count + 1
            # 显示画面
            cv.imshow("frame", frame)
            key = cv.waitKey(10)
            if key == 27:
                break
            elif key == 32:
                cv.waitKey(0)
                continue
        else:
            break
    video_cap.release()
    cv.destroyAllWindows()
    return frame_count, all_frames

def frame2gif(frames_list):
    imageio.mimsave('result.gif', frames_list, 'GIF', duration=0.08, loop=0)

if __name__ == '__main__':
    frame_count, all_frames = readVideo('./result.mp4')
    print(frame_count)
    frame2gif(all_frames)