CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 900 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/900 --lam 1.0 --xi 2.0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 800 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/800 --lam 1.0 --xi 3.0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 700 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/700 --lam 1.0 --xi 3.0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 600 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/600 --lam 1.0 --xi 5.0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 500 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/500 --lam 1.0 --xi 6.0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 400 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/400 --lam 1.0 --xi 9.0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 300 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/300 --lam 1.0 --xi 12.0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 200 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/200 --lam 1.0 --xi 16.0
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 100 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_s/100 --lam 1.0 --xi 20.0

