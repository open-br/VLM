import argparse
import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import sounddevice as sd

parser = argparse.ArgumentParser()
parser.add_argument('input_filename', help='input text file name')

args = parser.parse_args()

input_fname = args.input_filename
text2speech = Text2Speech.from_pretrained(
    train_config="../model/tts_male/exp/tts_male/config.yaml",
    model_file="../model/tts_male/exp/tts_male/50epoch.pth",
    vocoder_tag=str_or_none('none'),
    device="cuda"
)

pause = np.zeros(30000, dtype=np.float32)

with open(input_fname, 'r', encoding='utf-8') as f:
    sentences = f.readlines()
sentences = list(map(lambda s:s.rstrip("\n"), sentences))

duration = 2
while True:
    for sentence in sentences:
        wav_list = []
        final_wav = []
        sentence_list = sentence.split('<pause>')
        
        for speech_sentence in sentence_list:
            if speech_sentence == "":
                continue
            with torch.no_grad():
                result = text2speech(speech_sentence)["wav"]
                wav_list.append(np.concatenate([result.view(-1).cpu().numpy(), pause]))

        final_wav = np.concatenate(wav_list)

        sd.play(final_wav, text2speech.fs)
        sd.wait()
        sd.sleep(int(duration * 1000))
