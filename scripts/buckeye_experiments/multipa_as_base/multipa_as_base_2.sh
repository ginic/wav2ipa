#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=24GB
#SBATCH -p gpu-preempt-highvram
#SBATCH -G 4
#SBATCH --constraint=vram40
#SBATCH --exclude=gpu058
#SBATCH --nodes=1
#SBATCH --time 3:00:00
#SBATCH -o %j_multipa_as_base_2.out
#SBATCH --mail-type END

batch_size=4
grad_acc=1
learning_rate=3e-4
model_dir=data/models/multipa_as_base_2

dataset_cache=dataset_cache
data_dir=data/buckeye


module load conda/latest
conda activate ./env_cuda124

python --version

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --base_model ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns \
    --train_seed 112 \
    buckeye --train_samples 4000 --val_samples 5605 