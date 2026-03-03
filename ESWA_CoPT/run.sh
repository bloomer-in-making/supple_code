

for data in 'cb' 'copa' 'wsc' 'wic' 'rte' 'boolq'; do
for method in copt; do
for num in 1 ; do

model=t5-base

CUDA_VISIBLE_DEVICES=0 python main.py --datasets=${data} --model_name=${model} --enc_prompt_tokens 100 -ts 16 -e 100 --bottle_neck 10 --save_name ${model}_${data}_${method} --method ${method}

done
done
done









