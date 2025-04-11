import os
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
from text.cleaner import clean_text
import traceback
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
txt_path = "%s/2-name2text-%s.txt" % (opt_dir, gpu_id)

def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),gpu_id)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))


if os.path.exists(txt_path) == False:
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text, word2ph):
        ### input:
        ##  text:文本，word2ph音素数量
        ##  output: [hidden_dim, sum(word2ph)]
        ###
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt") # 文本分词，将 text 编码成bert模型所需的输入张量
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1] # 得到一个形状为[len(text), hidden_dim]的张量，表示每个字的bert向量

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1) # 对每个字，重复word2ph[i]次，相当于将特征扩展到音素层级
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T # 返回 [hidden_dim, sum(word2ph)]


    def process(str_info_list, str_info_post_list):
        for id, text, lang in str_info_list:
            try:
                # import pdb; pdb.set_trace()
                print(id)
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lang, version)
                path_bert = "%s/%s.pt" % (bert_dir, id)
                if os.path.exists(path_bert) == False and lang == "zh":
                    bert_feature = get_bert_feature(norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    # torch.save(bert_feature, path_bert)
                    my_save(bert_feature, path_bert)
                phones = " ".join(phones)
                str_info_post_list.append([id, phones, word2ph, norm_text])
            except:
                print(id, text, traceback.format_exc())

    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "KO": "ko",
        "Ko": "ko",
        "ko": "ko",
        "yue": "yue",
        "YUE": "yue",
        "Yue": "yue",
    }
    for line in lines[int(gpu_id) :: int(total_gpu)]:
        try:
            wav_path, spk_name, language, text = line.split("|")
            id = os.path.basename(wav_path)[:-4]
            # todo.append([name,text,"zh"])
            if language in language_v1_to_language_v2.keys():
                todo.append(
                    [id, text, language_v1_to_language_v2.get(language, language)]
                )
            else:
                print(f"\033[33m[Waring] The {language = } of {id} is not supported for training.\033[0m")
        except:
            print(line, traceback.format_exc())
    process(todo, res)
    opt = []
    for id, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (id, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")