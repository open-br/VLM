import cv2
import time

import argparse

def ArgParser():
    parser = argparse.ArgumentParser(description="description of the program")
    parser.add_argument("-t", "--time", type=int, default=30, help="capture loop time")
    parser.add_argument("-i", "--camera_id", type=int, default=0, help="camera id")
    return parser.parse_args() 

def main(cap, picture_time):
    # キャプチャがオープンしている間続ける
    past_time = time.perf_counter()
    while(cap.isOpened()):
        # フレームを読み込む
        ret, frame = cap.read()
        if ret == True:
            # フレームを表示
            cv2.imshow('Webcam Live', frame)
            if picture_time < (time.perf_counter() - past_time):
                cv2.imwrite('./imgs/tmp.jpg', frame)
                past_time = time.perf_counter()
            # 'q'キーが押されたらループから抜ける
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # キャプチャをリリースし、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = ArgParser()
    cap = cv2.VideoCapture(args.camera_id)
    main(cap, args.time)
    cap.release()
    cv2.destroyAllWindows()