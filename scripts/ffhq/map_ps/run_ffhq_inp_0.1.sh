# 全部重新调
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps/0 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 0
CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps/25 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 25
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps/50 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 50
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps/75 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 75
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps/100 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 100


CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/test_time --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 0
