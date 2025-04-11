import os
import sys
from glob import glob
import pdb

# dataset_type = "aishell-1"
# audio_root = "/home/bhc/BHC/Data/data_aishell/wav/"
# orignal_transciption = "/home/bhc/BHC/Data/data_aishell/transcript/aishell_transcript_v0.8.txt"
# language = "zh"
is_merge = True
dataset_type = sys.argv[1]
audio_root = sys.argv[2]
orignal_transciption = sys.argv[3]
language = sys.argv[4]
out_dir = sys.argv[5]

os.makedirs(out_dir, exist_ok=True)

if dataset_type == "aishell-1":
    dict_id_text = {}
    with open(orignal_transciption, "r") as f:
        lines = f.readlines()
        for line in lines:
            pos = line.find(" ")
            id = line[:pos]
            text = line[pos+1:] #这里暂时先不去除空格
            dict_id_text[id] = text
    types = ["train", "test", "dev"]
    str_info_dict = {"train": [], "test": [], "dev": []}
    new_id_start =0
    for type in types:
        sid_list = os.listdir(f"{audio_root}/{type}")
        for new_id, sid in enumerate(sid_list):
            audio_paths = glob(os.path.join(audio_root, f"{type}/{sid}/*.wav"))
            for path in audio_paths:
                try:
                    id = path.split("/")[-1][:-4]
                    text = dict_id_text[id][:-1].lstrip()
                    str_info = f"{path}|{new_id+new_id_start}|{language}|{text}"
                    str_info_dict[type].append(str_info)
                except:
                    print(f"not exist:{id}")
                    continue
        new_id_start += len(sid_list)
    if is_merge:
        str_info_list = []
        for k, l in str_info_dict.items():
            str_info_list += l
        with open("transcript.txt", "wt", encoding="utf-8") as f:
            for str_info in str_info_list:
                f.write(str_info + "\n")
    else:
        for key, l in str_info_dict.items():
            with open(f"{key}_metadata.txt", "wt", encoding="utf-8") as f:
                for str_info in l:
                    f.write(str_info + "\n")
elif dataset_type == "single_dir":
    dict_id_text = {}
    # import pdb; pdb.set_trace()
    with open(orignal_transciption, "r") as f:
        lines = f.readlines()
        for line in lines:
            pos = line.find(" ")
            id = line[:pos]
            text = line[pos+1:] #这里暂时先不去除空格
            dict_id_text[id] = text
        audio_paths = glob(os.path.join(audio_root, "*.wav"))
        l = []
        for path in audio_paths:
            try:
                id = path.split("/")[-1][:-4]
                text = dict_id_text[id][:-1].lstrip()
                str_info = f"{path}|0|{language}|{text}"
                l.append(str_info)
            except:
                print(f"not exist:{id}")
                continue
        with open(f"{out_dir}/transcript.txt", "wt") as f:
            f.write("\n".join(l))
        