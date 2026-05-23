# CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo reddiff --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/reddiff/0 --lam 0.1 --xi 1.0


CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo only_ps --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/mmse/25 --lam 0.1 --xi 1.0 --renoise_t 25
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo only_ps --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/mmse/50_seed_2026 --lam 0.1 --xi 1.0 --renoise_t 50 --seed 2026
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo only_ps --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/mmse/75 --lam 0.1 --xi 1.0 --renoise_t 75
CUDA_VISIBLE_DEVICES=3 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo only_ps --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/mmse/100 --lam 0.1 --xi 1.0 --renoise_t 100

