import requests
import torch
import transformers
from PIL import Image

from transformers.generation.streamers import TextStreamer, TextIteratorStreamer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llava.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llava.train.dataset import tokenizer_image_token
import time
import argparse
from threading import Thread
import cv2
import queue
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import sounddevice as sd
import pykakasi

import warnings
warnings.simplefilter('ignore')


def ArgParser():
    parser = argparse.ArgumentParser(description="description of the program")
    parser.add_argument("-g", "--gpu", action="store_true", default=True, help="enable CPU")
    parser.add_argument("-i", "--image", type=str, default="./imgs/tmp.jpg", help="image file")
    parser.add_argument("-c", "--camera", action="store_true", default=False, help="camera")
    parser.add_argument("--camera_id", type=int, default="0", help="camera id")
    parser.add_argument("-p", "--prompt", type=str, default="", help="prompt")
    parser.add_argument("-t", "--type", type=str, default="v1", help="type")
    parser.add_argument("--max_time", type=int, default="120", help="max_time")
    parser.add_argument("--max_sentence_len", type=int, default="128", help="max_sentence_len")
    parser.add_argument("--detail", action="store_true", default=False, help="output detail")
    parser.add_argument("--fast_streamer", action="store_true", default=False, help="another streamer")
    parser.add_argument("--ip_head", default="192.168.77.220",
        help="The ip of the OSC server")
    parser.add_argument("--port_head", type=int, default=22200,
        help="The port the OSC server is listening on")
    parser.add_argument("--ip_avater", default="127.0.0.1",
        help="The ip of the OSC server")
    parser.add_argument("--port_avater", type=int, default=5005,
        help="The port the OSC server is listening on")
    return parser.parse_args() 

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

def capture(camera_id):
    cap = cv2.VideoCapture(camera_id)

    # キャプチャがオープンしている間続ける
    while(cap.isOpened()):
        # フレームを読み込む
        ret, frame = cap.read()
        if ret == True:
            # フレームを表示
            cv2.imshow('Webcam Live', frame)

            # 'q'キーが押されたらループから抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('./imgs/tmp.jpg', frame)
                # time.sleep(0.5)
                break
        else:
            break

    # キャプチャをリリースし、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()
    return frame

def draw(im_list):
    #貼り付け
    plt.imshow(im_list)
    #表示
    # plt.show()
    # plt.pause(.01)
    plt.show()

def cut_sentence(input):
    if '。' in input:
        input = input[:input.rfind('。')+1]
    # print(input)
    return input

def main(args):
    if args.prompt != "":
        prompt = args.prompt
    else:
        prompt = "この画像の特徴を説明をしてください。"
    # image pre-process
    if args.camera:
        image = capture(args.camera_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_list = np.asarray(image)
        executor = ProcessPoolExecutor(max_workers=3)
        camera_future = executor.submit(draw, im_list)
    else:
        if args.image != "":
            # 手元の画像
            image = Image.open(args.image).convert('RGB')
        else:
            image_url = "https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg"
            image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
        im_list = np.asarray(image)
        executor = ProcessPoolExecutor(max_workers=3)
        camera_future = executor.submit(draw, im_list)

    max_time = args.max_time
    max_sentence_len = args.max_sentence_len

    total_time_start = time.perf_counter()
    # parser_hf = transformers.HfArgumentParser(
    #     (ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser_hf.parse_args_into_dataclasses()
    model_path = 'toshi456/llava-jp-1.3b-v1.1'
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    torch_dtype = torch.bfloat16 if device=="cuda" else torch.float32

    model = LlavaGpt2ForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="eager",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=1532,
        padding_side="right",
        use_fast=False,
    )
    model.eval()

    conv_mode = args.type
    conv = conv_templates[conv_mode].copy()

    image_size = model.get_model().vision_tower.image_processor.size["height"]
    if model.get_model().vision_tower.scales is not None:
        image_size = model.get_model().vision_tower.image_processor.size["height"] * len(model.get_model().vision_tower.scales)
    
    if device == "cuda":
        image_tensor = model.get_model().vision_tower.image_processor(
            image, 
            return_tensors='pt', 
            size={"height": image_size, "width": image_size}
        )['pixel_values'].half().cuda().to(torch_dtype)
    else:
        image_tensor = model.get_model().vision_tower.image_processor(
            image, 
            return_tensors='pt', 
            size={"height": image_size, "width": image_size}
        )['pixel_values'].to(torch_dtype)

    # create prompt
    # ユーザー: <image>\n{prompt}
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0)
    if device == "cuda":
        input_ids = input_ids.to(device)

    input_ids = input_ids[:, :-1] # </sep>がinputの最後に入るので削除する
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    # 通常のStreamer
    if args.fast_streamer != False:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=150.0)
        # predict
        predict_time_start = time.perf_counter()
        config = dict(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.1,
            top_p=1.0,
            max_new_tokens=max_sentence_len,
            streamer=streamer,
            use_cache=True,
            pad_token_id=4,
            length_penalty=10.0,
            early_stopping=True,
            max_time=max_time,
            num_return_sequences=1,
        )
        thread = Thread(target=model.generate, kwargs=config)
        thread.start()

        print("---推論開始---")
        generated_text = ""
        
        # ストリーミング処理
        text=buffer = ""
        for token in streamer:
            if token != '':
                # print(token)
                print(token, end="", flush=True)
                generated_text += token
            # if len(text_buffer)
    
    else:
        ## もしStreamerが遅い場合
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=100.0)

        # predict
        predict_time_start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.1,
                top_p=1.0,
                max_new_tokens=max_sentence_len,
                streamer=streamer,
                use_cache=True,
                pad_token_id=4,
                length_penalty=10.0,
                early_stopping=True,
                max_time=max_time,
            )
        output_ids = [token_id for token_id in output_ids.tolist()[0] if token_id != IMAGE_TOKEN_INDEX]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)

        target = "システム: "
        idx = output.find(target)
        input_text = output[:idx]
        generated_text = output[idx+len(target):]

    print("\n---推論終了---")
    predict_time_end = time.perf_counter()
    predict_time = predict_time_end-predict_time_start

    # 文章が途中で切れた場合、切れた部分を削除
    output_text = cut_sentence(generated_text)

    print('最終出力：', output_text)
    
    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start
    if args.detail:
        print('---詳細---')
        print('入力：', prompt)
        print('出力：', output_text)
        print('トークン数:{:d}'.format(len(output_text)))
        print('推論速度:{:.2f}[token/s]'.format(len(output_text)/predict_time))
        print('全体処理時間:{:.2f}[s]'.format(total_time))

    # TTS
    text2speech = Text2Speech.from_pretrained(
        train_config="./model/tts_male/exp/tts_male/config.yaml",
        model_file="./model/tts_male/exp/tts_male/50epoch.pth",
        vocoder_tag=str_or_none('none'),
        device="cuda"
    )

    pause = np.zeros(10000, dtype=np.float32)

    
    sentence_list = output_text.split('。')
    
    for speech_sentence in sentence_list:
        wav_list = []
        final_wav = []
        vowel_text = vowel(speech_sentence)

        if speech_sentence == "":
            continue
        with torch.no_grad():
            result = text2speech(speech_sentence)["wav"]
            wav_list.append(np.concatenate([result.view(-1).cpu().numpy(), pause]))

        final_wav = np.concatenate(wav_list)

        client_head  = udp_client.SimpleUDPClient(args.ip_head, args.port_head)
        # client_avater = udp_client.SimpleUDPClient(args.ip_avater, args.port_avater)
        # # メッセージ送信
        client_head.send_message("/firebarion/voice/command", [vowel_text,vowel_speed])
        # client_avater.send_message("/firebarion/voice/command ", vowel_text,vowel_speed) 
        sd.play(final_wav, text2speech.fs)
        sd.wait()
        # sd.sleep(int(duration * 1000))

    for process in executor._processes.values():
        process.kill()
    
if __name__ == "__main__":
    args = ArgParser()
    main(args)
