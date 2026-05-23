# 全部重新调
# CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps/0 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 0
# CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps/200 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 200
# CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps/400 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 400
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps/100 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 100
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps/300 --algo map_ps --xi 1.0 --optimize_iters 300 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 300
# CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_ps/150 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 150
# CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_ps/200 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 200

# python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/map_ps/0 --algo map_ps --xi 1.0 --optimize_iters 100 --vae_lr 0.5 --w_prior 0.25 --noise_t 10 --renoise_t 0
 

