# CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_a/1 --lam 1.0 --xi 1.0 --M 1
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_a/2 --lam 1.0 --xi 1.0 --M 2
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_a/4 --lam 1.0 --xi 1.0 --M 4
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_a/8 --lam 1.0 --xi 1.0 --M 8
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_a/16 --lam 1.0 --xi 1.0 --M 16
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/sr4_0.2/dps_a/32 --lam 1.0 --xi 1.0 --M 32


CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr4 --sigma_0 0.1 -i ffhq/test_time --lam 1.0 --xi 1.0 --M 2
