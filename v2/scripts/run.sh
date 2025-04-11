export PYTHONPATH=PYTHONPATH:.

start_stage=1
end_stage=1
s1_old_config=configs/s1longer-v2.yaml
s1_new_gpt=configs/new_gpt.yaml
s2_old_config=configs/s2.json
s2_new_sovits=configs/new_sovits.yaml
exp_root=logs
exp_name=my_exp
save_root=$exp_root/$exp_name
version=v2

SoVITS_weight_root=weights/SoVITS_weights,weights/SoVITS_weights_v2,weights/SoVITS_weights_v3
GPT_weight_root=weights/GPT_weights,weights/GPT_weights_v2,weights/GPT_weights_v3
IFS="," read -ra sovits_dirs <<< "$SoVITS_weight_root"
IFS="," read -ra gpt_dirs <<< "$GPT_weight_root"

for dir in "${sovits_dirs[@]}"; do
  mkdir -p $dir
done
for dir in "${gpt_dirs[@]}"; do
  mkdir -p $dir
done

if [ $start_stage -le 0 ] && [ $end_stage -ge 0 ]; then
    echo "GPT训练"
    python my_utils/update_config.py $s1_old_config $s1_new_gpt 1
    CUDA_VISIBLE_DEVICES=0,1 python s1_train.py --config_file TEMP/tmp_s1.yaml
fi

if [ $start_stage -le 1 ] && [ $end_stage -ge 1 ]; then
    echo "Sovits训练"
    python my_utils/update_config.py $s2_old_config $s2_new_sovits 2
    CUDA_VISIBLE_DEVICES=0 python s2_train.py --config TEMP/tmp_s2.json
fi