CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 900 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/900 --lam 1.0 --xi 2.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 800 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/800 --lam 1.0 --xi 2.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 700 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/700 --lam 1.0 --xi 3.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 600 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/600 --lam 1.0 --xi 3.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 500 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/500 --lam 1.0 --xi 5.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 400 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/400 --lam 1.0 --xi 6.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 300 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/300 --lam 1.0 --xi 9.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 200 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/200 --lam 1.0 --xi 13.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 100 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/dps_s/100 --lam 1.0 --xi 19.0

