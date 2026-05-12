#!/bin/bash
#SBATCH --job-name=zebrafish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=testlogs_aug/slurm_%j.out
#SBATCH --error=testlogs_aug/slurm_%j.err

cd $SLURM_SUBMIT_DIR

mkdir -p testlogs_aug

source ~/.bashrc
conda activate attn_unet_env

echo "start test model"
echo "Working directory: $(pwd)"

if [ $epoch_time ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=150
fi

if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_output_aug'
fi

if [ $cfg ]; then
    CFG=$cfg
else
    CFG='/projects/wan-lab/zebrafish/swin_unet_2d_implementation/Swim-unet_patch/configs/swin_tiny_patch4_window7_224_lite.yaml'
fi

if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='datasets/Synapse'
fi

if [ $learning_rate ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.05
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

if [ $batch_size ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=24
fi

echo "start test model"
# python test.py --dataset Synapse --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE

python /projects/wan-lab/zebrafish/unet_attn_2d_implementation/Swim-unet_patch/test.py  --is_saveni --cfg /projects/wan-lab/zebrafish/unet_attn_2d_implementation/Swim-unet_patch/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /projects/wan-lab/zebrafish/annotations_harshil/ --output_dir /projects/wan-lab/zebrafish/unet_attn_2d_implementation/Swim-unet_patch/model_output_aug