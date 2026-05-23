CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_ps_a/2 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100 --M 2
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_ps_a/4 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100 --M 4
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_ps_a/8 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100 --M 8
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_ps_a/16 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100 --M 16
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_ps_a/32 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100 --M 32


CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/test_time --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100 --M 2
