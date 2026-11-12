#!/bin/bash
# Train an audio to IPA transcription speech recognizer on Unity. This script is for quickly testing trainign with Unity, so it 
# requires lesser resources and training epochs than would be required to fully train a model.
# Note that as of September 2026 for full train jobs to succeed on Unity, you should use the following SBATCH options: 
# -p gpu-preempt-highvram (since3 we want at least 40GB vram) 
# -G 4 
# --constraint=vram40 
# --constraint=sm80 (to avoid nodes that don't support recent verions of Pytorch)


#SBATCH -c 8
#SBATCH -n 1
#SBATCH --mem=32GB
#SBATCH --constraint=vram16
#SBATCH -p gpu-preempt
#SBATCH -G 1 
#SBATCH --time 12:00:00
#SBATCH -o train_test_%j.out
#SBATCH --mail-type END

data_dir=data/buckeye
cache_dir=dataset_cache
vocab_file=data/vocab.en.json

model_dir=data/test_model

module load conda/latest

conda activate ./env_cuda124

# Wrapper script to train models 
multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 2 --num_proc 8 \
    --mask_time_length 9 \
    buckeye --train_samples 1000 --val_samples 200
