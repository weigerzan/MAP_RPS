CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps_a/2 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 300 --M 2
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps_a/4 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 300 --M 4
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps_a/8 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 300 --M 8
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps_a/16 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 300 --M 16
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps_a/32 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 300 --M 32


CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/test_time --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 300 --M 2
