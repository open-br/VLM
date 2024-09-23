@echo off
chcp 65001
call "C:\Users\firebarion\anaconda3\etc\profile.d\conda.sh"
call activate llava_jp
:LOOP
python mft2024_tts.py
goto :LOOP