import os
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
from text.cleaner import clean_text
from module.models import SynthesizerTrn
import traceback
import utils
from time import time as ttime
import torch
import shutil
bert_pretrained_dir = "pretrained_models/chinese-roberta-wwm-ext-large"

exp_root = sys.argv[1]
exp_name = sys.argv[2]
gpu_id = sys.argv[3]
total_gpu = sys.argv[4]
version = sys.argv[5]
inp_text = sys.argv[6]
is_half = True
opt_dir="%s/%s"%(exp_root,exp_name)

hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, gpu_id)
config = "configs/s2.json"


pretrained_sovits_name=["pretrained_models/s2G488k.pth", "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth","pretrained_models/s2Gv3.pth"]
pretrained_s2G = pretrained_sovits_name[int(version[-1])-1]


if os.path.exists(semantic_path) == False:
    os.makedirs(opt_dir, exist_ok=True)
    device = f"cuda:{gpu_id}"
    hps = utils.get_hparams_from_file(config)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version=version,
        **hps.model
    )
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(
        vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu", weights_only=False)["weight"], strict=False
        )
    )
    
    def get_token(wav_name, str_info_list):
        hubert_path = f"{hubert_dir}/{wav_name}.pt"
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half == True:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        str_info_list.append("%s\t%s" % (wav_name, semantic))


    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    for line in lines[int(gpu_id) :: int(total_gpu)]:
        try:
            wav_path, spk_name, language, text = line.split("|")
            id = os.path.basename(wav_path)[:-4]
            get_token(id, res)
        except:
            print(line, traceback.format_exc())
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(res))