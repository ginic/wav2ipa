#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH --time 12:00:00
#SBATCH -o %j_full_test_no_ffmpeg.out
#SBATCH --mail-type END

module load conda/latest
conda env create --prefix ./env_cuda124_noffmpeg --file=multipa_cuda124_no_ffmpeg.yml
conda activate ./env_cuda124_noffmpeg


conda list

python --version

pip install .[gpu,dev,test]

conda list

echo "How many GPUs found by pytorch?"
python -c "import torch; print(torch.cuda.device_count())"

python -m unidic download

# Data preprocessing
data_dir=data/new_buckeye
model_dir=data/models/brandon_virginia_no_ffmpeg
output_dir=data/virginia_testing_no_ffmpeg

batch_size=4
grad_acc=4
learning_rate=3e-4
rand_seed=29
train_samples=8000

echo "Training without ffmpeg"

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed $rand_seed \
    buckeye --train_samples $train_samples --val_samples 1000

echo "Evaluation without ffmpeg"

multipa-evaluate --local_models \
 $model_dir/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/full_dataset_train_3/wav2vec2-large-xlsr-53-buckeye-ipa \
 --eval_out $output_dir/buckeye_eval.csv \
 --verbose_results_dir $output_dir/detailed_results \
 --data_dir $data_dir --no_space --use_gpu --num_proc 8