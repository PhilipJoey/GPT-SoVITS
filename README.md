
### Install
``` python
### python 3.9
conda create -n gpt-sovits python=3.9
conda activate gpt-sovits
pip install -r requirements.txt

sudo apt install ffmpeg

```

### Download
1. Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `v2/pretrained_models`.
2. Chinese v2 additional: [G2PWModel_1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip)（Download G2PW models, unzip and rename to `G2PWModel`, and then place them in `GPT_SoVITS/text`.
### Infer
1. change file `v2/single_infer/args.yaml`
2. infer
``` python
### infer with pretrained model
cd v2
bash scripts/infer.sh # output save in 
```
3. output save in `single_infer`