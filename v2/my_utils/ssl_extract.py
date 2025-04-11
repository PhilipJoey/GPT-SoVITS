import os
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
from text.cleaner import clean_text
import sys, numpy as np, traceback, pdb
from time import time as ttime
import torch
import shutil
import ffmpeg
import librosa
from scipy.io import wavfile
from feature_extractor import cnhubert
cnhubert.cnhubert_base_path = "pretrained_models/chinese-hubert-base"

exp_root = sys.argv[1]
exp_name = sys.argv[2]
gpu_id = sys.argv[3]
total_gpu = sys.argv[4]
version = sys.argv[5]
inp_text = sys.argv[6]
is_half = False
opt_dir="%s/%s"%(exp_root,exp_name)
txt_path = "%s/2-name2text-%s.txt" % (opt_dir, gpu_id)

def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),gpu_id)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))


hubert_dir="%s/4-cnhubert"%(opt_dir)
wav32dir="%s/5-wav32k"%(opt_dir)
os.makedirs(opt_dir,exist_ok=True)
os.makedirs(hubert_dir,exist_ok=True)
os.makedirs(wav32dir,exist_ok=True)
device = f"cuda:{gpu_id}"
maxx=0.95
alpha=0.5
model=cnhubert.get_model()
if is_half == True:
    model = model.half().to(device)
else:
    model = model.to(device)

def get_ssl(id, wav_path):
    hubert_path = f"{hubert_dir}/{id}.pt"
    if os.path.exists(hubert_path): return
    out, _ = (
        ffmpeg.input(wav_path, threads=0)
        .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=32000)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    tmp_audio = np.frombuffer(out, np.float32).flatten()
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (id, tmp_max))
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )#不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()
    wavfile.write(
        "%s/%s"%(wav32dir,id),
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl,hubert_path)

with open(inp_text, "r", encoding="utf8") as f:
    lines = f.read().strip("\n").split("\n")

for line in lines[int(gpu_id) :: int(total_gpu)]:
    try:
        wav_path, spk_id, language, text = line.split("|")
        id = os.path.basename(wav_path)[:-4]
        get_ssl(id, wav_path)
    except:
        print(line, traceback.format_exc())