CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps_a/2 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 75 --M 2
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps_a/4 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 75 --M 4
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps_a/8 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 75 --M 8
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps_a/16 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 75 --M 16
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/map_ps_a/32 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 75 --M 32

CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/test_time --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.7 --noise_t 10 --renoise_t 75 --M 2
