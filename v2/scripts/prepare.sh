
export PYTHONPATH=PYTHONPATH:.
start_stage=2
end_stage=2

exp_root=logs
exp_name=my_exp
save_root=$exp_root/$exp_name
version=v2
data_path=/home/bhc/BHC/Data/aishell/S0094
inp_text=/home/bhc/BHC/Data/data_aishell/transcript/aishell_transcript_v0.8.txt

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $start_stage -le 0 ] && [ $end_stage -ge 0 ]; then
	echo "生成metadata文件"
	python my_utils/collect.py single_dir $data_path $inp_text zh $exp_root/$exp_name
	echo "Stage0 Done all!"
fi

if [ $start_stage -le 1 ] && [ $end_stage -ge 1 ]; then
	echo "文本分词与特征提取(音素, bert特征)"
	for ((i=0; i<gpu_num; i++)); do
		echo "启动任务在 GPU $i 上"
		python my_utils/make_tokenize.py $exp_root $exp_name $i $gpu_num v2 $exp_root/$exp_name/transcript.txt > log_$i 2>&1 &
	done
	wait
	cat $save_root/2-*.txt > $save_root/2-name2text.txtr
	rm $save_root/2*.txt
	mv $save_root/2-name2text.txtr $save_root/2-name2text.txt
	echo "Stage1 Done all!"
fi

if [ $start_stage -le 2 ] && [ $end_stage -ge 2 ]; then
	echo "语音自监督特征提取"
	for ((i=0; i<gpu_num; i++)); do
		echo "启动任务在 GPU $i 上"
		python my_utils/ssl_extract.py $exp_root $exp_name $i $gpu_num v2 $exp_root/$exp_name/transcript.txt > log_$i 2>&1 &
	done
	wait
	echo "Stage2 Done all!"
fi

if [ $start_stage -le 3 ] && [ $end_stage -ge 3 ]; then
	echo "语义token提取"
	# for ((i=0; i<gpu_num; i++)); do
	# 	echo "启动任务在 GPU $i 上"
	# 	python my_utils/token_extract.py $exp_root $exp_name $i $gpu_num v2 transcript.txt > log_$i 2>&1 &
	# done
	echo -e "item_name\tsemantic_audio" > $save_root/6-name2semantic.tsvr
	cat $save_root/6-*.tsv > $save_root/6-name2semantic.tsvr
	rm $save_root/6*.tsv
	mv $save_root/6-name2semantic.tsvr $save_root/6-name2semantic.tsv
	# wait
	echo "Stage3 Done all!"
fi