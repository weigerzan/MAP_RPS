CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i rebuttal/ffhq/sr4_0.1/ccdf/0 --algo ccdf --xi 0.1 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i rebuttal/ffhq/sr4_0.1/ccdf/100 --algo ccdf --xi 0.1 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100
CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i rebuttal/ffhq/sr4_0.1/ccdf/200 --algo ccdf --xi 0.1 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 200
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i rebuttal/ffhq/sr4_0.1/ccdf/300 --algo ccdf --xi 0.1 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 300
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i rebuttal/ffhq/sr4_0.1/ccdf/400 --algo ccdf --xi 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 400
 

