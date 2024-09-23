# Auto
デスクトップに配置してある該当バッチファイルをダブルクリック。
無限ループさせているので消すときはコマンドラインの×で消す
プログラムはデフォルトで３０秒間隔。変更する場合は中身のLOOP_TIMEを修正。
## 自己紹介
mft2024_speak.bat
## 環境説明（事前準備した写真）
mft2024_picture.bat
## 環境説明（Webカメラ）
### 先にカメラを起動しておく
cd worl/VLM
python auto_capture_camera.py

mft2024_online_camera.bat

# Manual
cd work/VLM
conda activate llava_jp
## 自己紹介
python mft2024_tts.py
## 環境説明（事前準備した写真）
python mft2024_camera.py -g -p この画像に写っている人の特徴を説明をしてください。 -i imgs/tmp.jpg
## 環境説明（Webカメラ）
python auto_capture_camera.py
python mft2024_camera.py -g -p この画像に写っている人の特徴を説明をしてください。 -i imgs/tmp.jpg
## 環境説明（手動でWebカメラ, カメラ画面で"q"を押したら撮影されて処理がはじまる）
python mft2024_camera.py -g -c -p この画像に写っている人の特徴を説明をしてください。

# その他
## 事前準備用の写真を撮りたいとき
cd example
python camera.py
## カメラが複数ついている場合、任意のIDを知りたいとき。または違うカメラをつかんだ時
python camera_check.py
id=0以外を使いたいときは引数-iに番号を渡せばよい。
関係するのはmft2024_online_camera.batのみなので、CAMERA_IDを書き換えること



#動作確認
python mft2024_camera.py -g -p 猫の隣には何がありますか？ -i imgs/sample.jpg
## 写真に対する説明文のみ出力
cd example
python camera.py
cd ../
python demo_llava_streaming.py -g -p 猫の隣には何がありますか？ -i imgs/sample.jpg
python demo_llava_streaming.py -g -p この画像に写っている人の特徴を説明をしてください。 -i imgs/tmp_0.jpg
python demo_llava_streaming.py -g -p このロボットの特徴を説明をしてください。 -i imgs/fiverion.jpg
python demo_llava_streaming.py -g -p このキャラクターの特徴を説明をしてください。 -i imgs/mario.jpg

python mft2024_camera.py -g -p 猫の隣には何がありますか？ -i imgs/sample.jpg
python mft2024_camera.py -g -p このキャラクターの特徴を説明をしてください。 -i imgs/mario.jpg

## どっちが早いか確認
python demo_llava.py -g -p 猫の隣には何がありますか？ -i imgs/sample.jpg
python demo_llava_streaming.py -g -p 猫の隣には何がありますか？ -i imgs/sample.jpg　--detail

# 口パク確認
cd example
python move_mouse.py
# OSC確認
python send_osc.py

