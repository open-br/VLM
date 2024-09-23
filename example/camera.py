import cv2
import os

import argparse

def ArgParser():
    parser = argparse.ArgumentParser(description="description of the program")
    parser.add_argument("-i", "--camera_id", type=int, default=0, help="camera id")
    return parser.parse_args() 

def main(cap):
    # キャプチャがオープンしている間続ける
    counter = 0
    while(cap.isOpened()):
        # フレームを読み込む
        ret, frame = cap.read()
        if ret == True:
            # フレームを表示
            cv2.imshow('Webcam Live', frame)

            # 'q'キーが押されたらループから抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('r'):
                print(cv2.imwrite('../imgs/tmp_' + str(counter) + '.jpg', frame))
                counter+=1
        else:
            break

    # キャプチャをリリースし、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = ArgParser()
    cap = cv2.VideoCapture(args.camera_id)
    main(cap)
    cap.release()
    cv2.destroyAllWindows()