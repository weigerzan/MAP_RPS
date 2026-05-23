# 全部重新调
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg cs2 --sigma_0 0.1 -i ffhq/cs2_0.1/vsdps/1.0 --lam 1.0 --xi 1.0
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg cs2 --sigma_0 0.1 -i ffhq/cs2_0.1/vsdps/0.8 --lam 0.8 --xi 1.5
CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg cs2 --sigma_0 0.1 -i ffhq/cs2_0.1/vsdps/0.5 --lam 0.5 --xi 6.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg cs2 --sigma_0 0.1 -i ffhq/cs2_0.1/vsdps/0.3 --lam 0.3 --xi 8.0
CUDA_VISIBLE_DEVICES=3 python main.py --ni --config ffhq.yml --timesteps 1000 --deg cs2 --sigma_0 0.1 -i ffhq/cs2_0.1/vsdps/0.0 --lam 0.0 --xi 9.0
