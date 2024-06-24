#!/bin/sh
env="battlefield"
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env},  algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_battlefield_team.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 40 --episode_length 50 \
    --num_env_steps 2000 --ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" \
    --user_name "yuchao" 
done