import requests
import torch
import transformers
from PIL import Image

from transformers.generation.streamers import TextStreamer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llava.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llava.train.dataset import tokenizer_image_token
import time
import argparse

import warnings
warnings.simplefilter('ignore')


def ArgParser():
    parser = argparse.ArgumentParser(description="description of the program")
    ## GPT Parameter
    parser.add_argument("-g", "--gpu", action="store_true", default=False, help="enable CPU")
    parser.add_argument("-i", "--image", type=str, default="", help="image file")
    parser.add_argument("-p", "--prompt", type=str, default="", help="prompt")
    parser.add_argument("-t", "--type", type=str, default="v1", help="prompt")
    return parser.parse_args() 


def main(args):
    if args.prompt != "":
        prompt = args.prompt
    else:
        prompt = "猫の隣には何がありますか？"
    # image pre-process
    if args.image != "":
        # 手元の画像
        image = Image.open(args.image).convert('RGB')
    else:
        image_url = "https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    
    total_time_start = time.perf_counter()
    parser_hf = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
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
    streamer = TextStreamer(tokenizer, skip_prompt=True, timeout=20.0)

    # predict
    predict_time_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.1,
            top_p=1.0,
            max_new_tokens=128,
            streamer=streamer,
            use_cache=False,
            pad_token_id=4,
            length_penalty=10.0,
            early_stopping=True,
            # max_time=60,
        )
    predict_time_end = time.perf_counter()
    predict_time = predict_time_end-predict_time_start

    output_ids = [token_id for token_id in output_ids.tolist()[0] if token_id != IMAGE_TOKEN_INDEX]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    target = "システム: "
    idx = output.find(target)
    input_text = output[:idx]
    output_text = output[idx+len(target):]
    
    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start
    print('---詳細---')
    print('入力：', input_text)
    print('出力：', output_text)
    print('トークン数:{:d}'.format(len(output_text)))
    print('推論速度:{:.2f}[token/s]'.format(len(output_text)/predict_time))
    print('全体処理時間:{:.2f}[s]'.format(total_time))
    
if __name__ == "__main__":
    args = ArgParser()
    main(args)