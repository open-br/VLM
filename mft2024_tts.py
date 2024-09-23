import argparse
import random
import time
from pythonosc import udp_client
import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import sounddevice as sd
import pykakasi

LOOP_TIME = 3.0

import warnings
warnings.simplefilter('ignore')

def ArgParser():
  parser = argparse.ArgumentParser(description="description of the program")
  parser.add_argument('--input_text', type=str, default="./mft2024_tts.txt", help='input text file name')
  parser.add_argument("--ip_head", default="192.168.8.197",
      help="The ip of the OSC server")
  parser.add_argument("--port_head", type=int, default=22220,
      help="The port the OSC server is listening on")
  parser.add_argument("--ip_avater", default="127.0.0.1",
    help="The ip of the OSC server")
  parser.add_argument("--port_avater", type=int, default=5005,
      help="The port the OSC server is listening on")
  return parser.parse_args() 

def speak(input_text):
  # input_text = args.input_text
  text2speech = Text2Speech.from_pretrained(
      train_config="./model/tts_male/exp/tts_male/config.yaml",
      model_file="./model/tts_male/exp/tts_male/50epoch.pth",
      vocoder_tag=str_or_none('none'),
      device="cuda"
  )
  # 1秒delay
  DELAY_TIME = 1.0
  pause = np.zeros(int(text2speech.fs*DELAY_TIME), dtype=np.float32)

  sentence_list = input_text.split('<pause>')
  wav_list = []

  for sentence in sentence_list:
      with torch.no_grad():
          result = text2speech(sentence)["wav"]
          wav_list.append(np.concatenate([pause, result.view(-1).cpu().numpy(), pause]))

  final_wav = np.concatenate(wav_list)
  # 発話の始終端にそれぞれ1秒のディレイが入っているため、２秒引く
  speak_time = len(final_wav)/text2speech.fs - 2.0
  # while True:
  #     sd.play(final_wav, text2speech.fs)
  #     sd.wait()
  return speak_time, final_wav, text2speech.fs

def convert_word_type_text(text, word_type):
  kakasi = pykakasi.kakasi()
  # 文章をkakasiに突っ込んで、単語ごとに分解されたリストを取得(単語は辞書型で色々な形式に変換された状態で格納されている)
  word_list = kakasi.convert(text)
  word_list_len = len(word_list)
  word_type_word_list = [word_list[cnt][word_type] for cnt in range(word_list_len)]
  word_type_text = " ".join(word_type_word_list)
  return word_type_text

def convert_special_word_to_vowel(text):
  nyanyunyo_vowel_dict =  {'にゃ':'あ', 'にゅ':'う', 'にょ':'お'}
  for key, value in nyanyunyo_vowel_dict.items():
    text = text.replace(key, value)
  return text

def extract_vowel(text):
  # ローマ字に対応するひらがなの母音と"ん"の辞書
  hiragana_vowel_and_n_dict = {'a':'あ','i':'い','u':'う','e':'え','o':'お','n':'ん'}
  hiragana_vowel_dict = hiragana_vowel_and_n_dict.copy()
  hiragana_vowel_dict.pop('n')
  vowel_and_n_text = ""
  word_cnt = 0
  for word in text:
    word_cnt += 1
    if (word in hiragana_vowel_dict) or ((word == 'n') and ((len(text) == word_cnt) or (text[word_cnt] == "'") or (text[word_cnt] not in hiragana_vowel_dict))):
    #   vowel_and_n_text += hiragana_vowel_and_n_dict[word]
      vowel_and_n_text += word
    else:
      pass
  return vowel_and_n_text

def vowel(input_text):
  hiragana_text = convert_word_type_text(input_text, 'hira')
  hiragana_text = convert_special_word_to_vowel(hiragana_text)
  # print(hiragana_text)
  latin_alpahbet_text = convert_word_type_text(hiragana_text, 'kunrei')
  # print(latin_alpahbet_text)
  # vowel_text = list(extract_vowel(latin_alpahbet_text))
  vowel_text = extract_vowel(latin_alpahbet_text)
  # print(vowel_text)
  return vowel_text

def main(args):
  
  with open(args.input_text, "r", encoding="utf-8") as f:
    for line in f:
      input_text = line.strip()
      
      if input_text == '<start>':
        continue
      elif input_text == '<end>':
        # time.sleep(LOOP_TIME)
        continue
      elif input_text == '':
        continue
      print(input_text)
      if 'MFT2024' in input_text:
        input_text = input_text.replace('MFT2024', 'メイカーフェア東京2024')
      # print(input_text)
      vowel_text = vowel(input_text)
      speak_time, speech, fs = speak(input_text)
      # print(vowel_text)
      # print(speak_time, "[s]")
      sd.play(speech, fs)
      client_head  = udp_client.SimpleUDPClient(args.ip_head, args.port_head)
      # client_avater = udp_client.SimpleUDPClient(args.ip_avater, args.port_avater)
      # # メッセージ送信
      client_head.send_message("/firebarion/voice/command", [vowel_text,speak_time])
      # client_avater.send_message("/firebarion/voice/command ", input_text,speak_time) 
      sd.wait()

if __name__ == "__main__":
    args = ArgParser()
    main(args)