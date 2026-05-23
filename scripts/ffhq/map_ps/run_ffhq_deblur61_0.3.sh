# 全部重新调
# CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.3 -i ffhq/deblur_gaussian_61_0.3/map_ps/0 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 0
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.3 -i ffhq/deblur_gaussian_61_0.3/map_ps/test --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 25
# CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.3 -i ffhq/deblur_gaussian_61_0.3/map_ps/50 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 50
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.3 -i ffhq/deblur_gaussian_61_0.3/map_ps/75 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 75
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.3 -i ffhq/deblur_gaussian_61_0.3/map_ps/100 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100
# CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.3 -i ffhq/deblur_gaussian_61_0.3/map_ps/150 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 150
# CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.3 -i ffhq/deblur_gaussian_61_0.3/map_ps/200 --algo map_ps --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 200
 

