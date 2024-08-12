#!/bin/bash
# Created by yuanding on 2022/07/20
# 使用CTC/AED模型解码脚本

set -e
. ./path.sh || exit 1

gpus="0"
export CUDA_VISIBLE_DEVICES=$gpus

stage=1
stop_stage=2

data_type=raw
num_utts_per_shard=1000

test_sets="test"
#"test_1000Cantonese test_magicdatacantonese test_commonvoicecantonese test_datatang500h"
#"test_1000Cantonese test_magicdatacantonese test_commonvoicecantonese test_datatang500h"
#test_1000Cantonese 
average_num=5
average_checkpoint=false
decode_modes="attention_rescoring"

. tools/parse_options.sh || exit 1

if [[ ! $# -eq 3 ]]; then
  echo "Usage: $0 <model_path> <data_dir> <result_dir>"
  echo "exp_dir: 模型实验文件夹, 一般为exp/unified_conformer."
  echo "data_dir: 数据文件夹, 一般为data, 需要包含dict文件."
  echo "result_dir: 结果文件夹."
  echo "--average_checkpoint: 默认true."
  echo "--average_num: 默认5."
  exit 1
fi

dir=$1
data_dir=$2
dict=$2/dict/lang_char.txt
result_dir=$3

if [[ ${average_checkpoint} == "true" ]]; then
  decode_checkpoint=$dir/avg_${average_num}.pt
else
  decode_checkpoint=$dir/lora_init.pt
  #$dir/init.pt
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: 准备基础数据."
  for x in ${test_sets}; do
    python3 -m local.prepare_data.prepare_test_data $x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: 生成指定格式的数据."
  for x in ${test_sets}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 $data_dir/$x/wav.scp $data_dir/$x/text --resample 16000 \
        $(realpath $data_dir/$x/shards) $data_dir/$x/data.list
    else
      tools/make_raw_list.py $data_dir/$x/wav.scp $data_dir/$x/text \
        $data_dir/$x/data.list
    fi
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$(date) stage 2: 解码."
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=-1
  ctc_weight=1.0
  reverse_weight=0
  for mode in ${decode_modes}; do
  {
    echo "当前解码模式: "${decode_modes}
    for x in ${test_sets}; do
    {
      test_name=test_${mode}${decoding_chunk_size:+_chunk$decoding_chunk_size}
      test_dir=$result_dir/${x}
      mkdir -p $test_dir
      echo "正在解码: ${x}, 解码日志: ${test_name}_${x}.log"
      python wenet/bin/recognize.py \
        --gpu $gpus \
        --modes $mode \
        --config $dir/train.yaml \
        --data_type $data_type \
        --test_data data/$x/data.list \
        --checkpoint $decode_checkpoint \
        --beam_size 10 \
        --batch_size 8 \
        --blank_penalty 0.0 \
        --ctc_weight $ctc_weight \
        --result_dir $test_dir/ \
        --use_lora true \
        --lora_ckpt_path $dir/epoch_9.pt \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
        >${test_name}_${x}.log 2>&1
      python tools/compute-wer.py --char=1 --v=1 \
        data/$x/text $test_dir/$decode_modes/text >$test_dir/$decode_modes/wer
    }
    done
#        --use_lora true \
# --lora_ckpt_path /root/autodl-tmp/wenet_origin/wenet/examples/aishell/s0/base_model_opt/epoch_1.pt \
  }
  done
fi
