CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 200 --deg denoise --algo dmap --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/dmap/0 --xi 12.0



CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo only_ps --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/dmap/25_0.5 --lam 0.1 --xi 0.5 --renoise_t 25
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo only_ps --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/dmap/50_0.5 --lam 0.1 --xi 0.5 --renoise_t 50
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo only_ps --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/dmap/75_0.5 --lam 0.1 --xi 0.5 --renoise_t 75
CUDA_VISIBLE_DEVICES=3 python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --algo only_ps --sigma_0 0.3 -i rebuttal/ffhq/denoise_0.3/dmap/100_0.5 --lam 0.1 --xi 0.5 --renoise_t 100
