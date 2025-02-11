
bs=64
ws=2
text_encoder=DecisionNCE-V
cuda=0

export MUJOCO_GL="osmesa"
export CUDA_VISIBLE_DEVICES=$cuda
export WANDB_API_KEY= # your wandb key


for seed in 0; do
file_name=libero_goal_seed$seed

torchrun --standalone --nnodes=1 --nproc_per_node=1 train_scripts/train_libero.py \
    --dataset_name 'libero_goal' \
    --json_path <your json path> \
    --algo_name 'robo_mutual' \
    --ddp False \
    --img_size 128 \
    --seed $seed \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --text_encoder $text_encoder \
    --ft_vision True \
    --film_fusion False \
    --ac_num 6 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 9 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0003 \
    --val_freq 10000000 \
    --eval_freq 50000 \
    --resume None \
    --wandb True \
    --steps 50000 \
    --save True \
    --save_freq 50000 \
    --T 25 \
    --save_path experiments/libero/reproduction/$file_name \
    --log_path experiments/libero/reproduction/$file_name \
    --port 2077 \
    --add_noise True \
    --minus_mean True \
    --mean_data_path ../config/mean_ood_7800ep.npz \
    --noise_data_path ../config/noise_id_7800ep.npz \
    --lang_prop '' \
    --cos_noise 0.2 \
    --cos_noise_decay 0.8 \
    --manual_img_goal False \
    --cut_path config/topk_index_80.npz
done