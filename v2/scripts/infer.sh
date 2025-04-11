export PYTHONPATH=PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0,1 python my_utils/inference.py v2 pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt \
        pretrained_models/gsv-v2final-pretrained/s2G2333k.pth pretrained_models/chinese-hubert-base \
        pretrained_models/chinese-roberta-wwm-ext-large False zh single_infer/args.yaml
