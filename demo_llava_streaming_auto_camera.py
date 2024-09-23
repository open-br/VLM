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

import warnings
warnings.simplefilter('ignore')


def ArgParser():
    parser = argparse.ArgumentParser(description="description of the program")
    parser.add_argument("-g", "--gpu", action="store_true", default=False, help="enable CPU")
    parser.add_argument("-i", "--image", type=str, default="", help="image file")
    parser.add_argument("-c", "--camera", action="store_true", default=False, help="camera")
    parser.add_argument("-p", "--prompt", type=str, default="", help="prompt")
    parser.add_argument("-t", "--type", type=str, default="v1", help="type")
    parser.add_argument("--max_time", type=int, default="120", help="max_time")
    parser.add_argument("--max_sentence_len", type=int, default="128", help="max_sentence_len")
    parser.add_argument("--detail", action="store_true", default=False, help="output detail")
    return parser.parse_args() 

def capture():
    cap = cv2.VideoCapture(0)

    # キャプチャがオープンしている間続ける
    while(cap.isOpened()):
        # フレームを読み込む
        ret, frame = cap.read()
        if ret == True:
            # フレームを表示
            cv2.imshow('Webcam Live', frame)

            # 'q'キーが押されたらループから抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
        prompt = "猫の隣には何がありますか？"
    # image pre-process
    if args.camera:
        image = capture()
    else:
        if args.image != "":
            # 手元の画像
            image = Image.open(args.image).convert('RGB')
        else:
            image_url = "https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg"
            image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
        # cv_image = cv2.imread(args.image)
        # cv2.imshow('Processing Picture', cv_image)
        im_list = np.asarray(image)
        # #貼り付け
        # plt.imshow(im_list)
        # #表示
        # # plt.show()
        # # plt.pause(.01)
        # plt.show(block=False)
        # plt.pause(3)
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
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=100.0)
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
    
    print("\n---推論終了---")
    predict_time_end = time.perf_counter()
    predict_time = predict_time_end-predict_time_start

    # 文章が途中で切れた場合、切れた部分を削除
    output_text = cut_sentence(generated_text)

    
    print('---最終出力---\n', output_text)
    
    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start
    if args.detail:
        print('---詳細---')
        print('入力：', prompt)
        print('出力：', output_text)
        print('トークン数:{:d}'.format(len(output_text)))
        print('推論速度:{:.2f}[token/s]'.format(len(output_text)/predict_time))
        print('全体処理時間:{:.2f}[s]'.format(total_time))

    for process in executor._processes.values():
        process.kill()
    
if __name__ == "__main__":
    args = ArgParser()
    main(args)
