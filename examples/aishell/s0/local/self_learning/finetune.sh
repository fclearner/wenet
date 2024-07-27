#!/bin/bash
# Created by yuanding on 2022/08/03
# CTC/AED模型调优训练

. ./path.sh || exit 1

stage=0
stop_stage=2
# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1
# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0

data_type=raw
num_utts_per_shard=1000
train_set=train
dev_set=dev
shards_dir=/home/wenet/examples/aishell/s0/data/shard/

cmvn=true

# use average_checkpoint will get better result
average_checkpoint=true
average_num=5
bpe_model=
gpus="0"
lr=0.0004
batch_size=8
epoch=20
warmup_steps=500
accum_grad=1
cpus=-1

train_engine=deepspeed
prefetch=500

deepspeed_config=conf/ds_stage1.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <data_dir> <model_dir> <out_dir>"
  echo "data_dir: 调优数据文件夹, 需要包含train和dev."
  echo "model_dir: 发版模型文件夹, 需包含self_learning文件夹, 部分旧模型不支持."
  echo "out_dir: 调优模型保存路径."
  echo "--average_num: 默认5."
  echo "--gpus: 显卡编号, ','连接, 如'0,1,2,3'."
  echo "--cpus: cpu训练时指定使用的cpu数量, 不指定则默认使用机器cpu核数的一半."
  echo "--lr: 学习率, 默认0.0004."
  echo "--batch_size: 默认16."
  echo "--epoch: 默认20."
  echo "--warmup_steps: 默认500"
  echo "--accum_grad: 默认2"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$gpus
data_dir=$1
model_dir=$2
out_dir=$3

self_learning=$model_dir/self_learning
#$model_dir/self_learning

train_config=$self_learning/exp/train.yaml
#conf/finetune_whisper_largev3_conv2d4_onlyattn.yaml
#conf/finetune_whisper_largev3_onlyattn.yaml
#conf/finetune_whisper_largev3_conv2d4_onlyattn.yaml
#$self_learning/train.yaml
# dict=data/dict/units.txt 
checkpoint=$self_learning/exp/init.pt
#$out_dir/step_19999.pt
#$self_learning/wenet_w2vbert_conformer_600m.pt
#$out_dir/epoch_4.pt
#$out_dir/epoch_9.pt
#$self_learning/wenet_whisper.remove-subsample.init-ctc.pt
#$out_dir/epoch_16.pt
#$self_learning/wenet_whisper.remove-subsample.init-ctc.pt
cmvn_dir=$self_learning/exp/global_cmvn

if [ -f $self_learning/data/lang_char/bpe.model ]; then
  bpe_model=$self_learning/data/lang_char/bpe.model
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: 生成指定格式的数据."
  for x in ${train_set} ${dev_set}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 data/$x/wav.scp data/$x/text --resample 16000 \
        $shards_dir data/$x/data.list
    else
      tools/make_raw_list.py data/$x/wav.scp data/$x/text \
        data/$x/data.list
    fi
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: 开始训练."
  mkdir -p $out_dir
  # INIT_FILE is for DDP synchronization
  INIT_FILE=$out_dir/ddp_init
  if [ -f $INIT_FILE ]; then
    rm $INIT_FILE
  fi
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  # NOTE(xcsong): deepspeed fails with gloo, see
  #   https://github.com/microsoft/DeepSpeed/issues/2818
  dist_backend="nccl"
  export NCCL_P2P_LEVEL=NVL
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  # NOTE(xcsong): Both ddp & deepspeed can be launched by torchrun
  # NOTE(xcsong): To unify single-node & multi-node training, we add
  #               all related args. You should change `nnodes` &
  #               `rdzv_endpoint` for multi-node, see
  #               https://pytorch.org/docs/stable/elastic/run.html#usage
  #               https://github.com/wenet-e2e/wenet/pull/2055#issuecomment-1766055406
  #               `rdzv_id` - A user-defined id that uniquely identifies the worker group for a job.
  #                           This id is used by each node to join as a member of a particular worker group.
  #               `rdzv_endpoint` - The rendezvous backend endpoint; usually in form <host>:<port>.
  # NOTE(xcsong): In multi-node training, some clusters require special NCCL variables to set prior to training.
  #               For example: `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=enp` + `NCCL_DEBUG=INFO`
  #               without NCCL_IB_DISABLE=1
  #                   RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL Version xxx
  #               without NCCL_SOCKET_IFNAME=enp  (IFNAME could be get by `ifconfig`)
  #                   RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:xxx
  #               ref: https://github.com/google/jax/issues/13559#issuecomment-1343573764
  # world_size=$(expr $num_gpus \* $num_nodes)
  # echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp ${cmvn_dir} ${out_dir}
  $cmvn && cmvn_opts="--cmvn ${out_dir}/global_cmvn"
  
  # CPU训练时的相关配置
  cp wenet/bin/train.py local/self_learning/
  if [ ${num_gpus} -eq 0 ]; then
    num_gpus=1
    world_size=-1
    if [ ${cpus} -ne -1 ]; then
      sed -i "/def main():/a\    torch.set_num_threads(${cpus})" local/self_learning/train.py
    fi
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"

  # export NCCL_DEBUG=INFO
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
          --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
  local/self_learning/train.py \
    --train_engine ${train_engine} \
    --config $train_config \
    --data_type $data_type \
    ${bpe_model:+--bpe_model $bpe_model} \
    --train_data $data_dir/${train_set}/data.list \
    --cv_data $data_dir/$dev_set/data.list \
    ${checkpoint:+--checkpoint $checkpoint} \
    --model_dir $out_dir \
    --ddp.dist_backend $dist_backend \
    --timeout 300 \
    --deepspeed_config ${deepspeed_config} \
    --deepspeed.save_states ${deepspeed_save_states} \
    --num_workers 1 \
    --prefetch ${prefetch} \
    --pin_memory
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  if [ ${average_checkpoint} == true ]; then
      decode_checkpoint=$out_dir/avg_${average_num}.pt
      echo "do model average and final checkpoint is $decode_checkpoint"
      python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $out_dir \
        --num ${average_num} \
        --val_best
  fi
fi
