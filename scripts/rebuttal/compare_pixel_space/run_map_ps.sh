CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg inp --sigma_0 0.05 -i rebuttal/ffhq/inp_0.05/map_ps/0 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.35 --noise_t 10 --renoise_t 0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg deblur_aniso --sigma_0 0.05 -i rebuttal/ffhq/deblur_aniso_0.05/map_ps/0 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.03 --noise_t 10 --renoise_t 0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg sr4 --sigma_0 0.05 -i rebuttal/ffhq/sr4_0.05/map_ps/0 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.1 --noise_t 10 --renoise_t 0



CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg sr4 --sigma_0 0.05 -i rebuttal/ffhq/sr4_0.05/map_ps/300 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.1 --noise_t 10 --renoise_t 300 --ps_method ddnm
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg deblur_aniso --sigma_0 0.05 -i rebuttal/ffhq/deblur_aniso_0.05/map_ps/300 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.03 --noise_t 10 --renoise_t 300 --ps_method ddnm
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg inp --sigma_0 0.05 -i rebuttal/ffhq/inp_0.05/map_ps/300 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.35 --noise_t 10 --renoise_t 300 --ps_method ddnm

CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg sr4 --sigma_0 0.05 -i rebuttal/ffhq/sr4_0.05/map_ps/100 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.1 --noise_t 10 --renoise_t 100 --ps_method ddnm
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg deblur_aniso --sigma_0 0.05 -i rebuttal/ffhq/deblur_aniso_0.05/map_ps/100 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.03 --noise_t 10 --renoise_t 100 --ps_method ddnm
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg inp --sigma_0 0.05 -i rebuttal/ffhq/inp_0.05/map_ps/100 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.35 --noise_t 10 --renoise_t 100 --ps_method ddnm




CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg sr4 --sigma_0 0.05 -i rebuttal/ffhq/sr4_0.05/map_ps/400 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.1 --noise_t 10 --renoise_t 400 --ps_method ddnm


CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 200 --deg inp --sigma_0 0.05 -i rebuttal/ffhq/inp_0.05/map_ps/test --algo map_ps --xi 0.1 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.1 --noise_t 10 --renoise_t 300 --ps_method ddnm


CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 200 --deg inp --sigma_0 0.05 -i rebuttal/ffhq/inp_0.05/map_ps/300_test_ddnm_200_0.15 --algo map_ps --xi 0.1 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.1 --noise_t 10 --renoise_t 300 --ps_method ddnm

CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 200 --deg sr4 --sigma_0 0.05 -i rebuttal/ffhq/sr4_0.05/map_ps/300_0.8_200 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.1 --noise_t 10 --renoise_t 300 --ps_method ddnm
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config ffhq.yml --timesteps 200 --deg deblur_aniso --sigma_0 0.05 -i rebuttal/ffhq/deblur_aniso_0.05/map_ps/300_0.8_200 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.03 --noise_t 10 --renoise_t 300 --ps_method ddnm
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 200 --deg inp --sigma_0 0.05 -i rebuttal/ffhq/inp_0.05/map_ps/300_0.8_200 --algo map_ps --xi 1.0 --optimize_iters 400 --vae_lr 0.5 --w_prior 0.35 --noise_t 10 --renoise_t 300 --ps_method ddnm
