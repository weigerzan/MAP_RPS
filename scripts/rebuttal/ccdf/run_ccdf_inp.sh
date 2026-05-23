CUDA_VISIBLE_DEVICES=3 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inpainting --sigma_0 0.1 -i rebuttal/ffhq/inpainting_0.1/ccdf/0 --algo ccdf --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 0
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inpainting --sigma_0 0.1 -i rebuttal/ffhq/inpainting_0.1/ccdf/25 --algo ccdf --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 25
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inpainting --sigma_0 0.1 -i rebuttal/ffhq/inpainting_0.1/ccdf/50 --algo ccdf --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 50
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inpainting --sigma_0 0.1 -i rebuttal/ffhq/inpainting_0.1/ccdf/75 --algo ccdf --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 75
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inpainting --sigma_0 0.1 -i rebuttal/ffhq/inpainting_0.1/ccdf/100 --algo ccdf --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100
 

