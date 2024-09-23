import cv2

#解説2
def get_available_cameras(max_devices = 10):
    # この関数は最初のmax_devicesデバイスをチェックし、利用可能なデバイスのリストを返します。
    available_devices = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print('警告: ビデオソースを開けません: ', i)
        else:
            print('成功: ビデオソースを見つけました: ', i)
            available_devices.append(i)
        cap.release()  # デバイスを開放します。
    return available_devices

if __name__ == "__main__":
    get_available_cameras()