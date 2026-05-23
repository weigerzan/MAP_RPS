
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr_bicubic8 --sigma_0 0.0 -i ffhq/sr8/vsdps/1.0 --lam 1.0 --xi 10.0
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr_bicubic8 --sigma_0 0.0 -i ffhq/sr8/vsdps/0.8 --lam 0.8 --xi 12.0
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr_bicubic8 --sigma_0 0.0 -i ffhq/sr8/vsdps/0.5 --lam 0.5 --xi 24.0
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr_bicubic8 --sigma_0 0.0 -i ffhq/sr8/vsdps/0.3 --lam 0.3 --xi 24.0
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr_bicubic8 --sigma_0 0.0 -i ffhq/sr8/vsdps/0.0 --lam 0.0 --xi 26.0
# CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg sr_bicubic8 --sigma_0 0.0 -i ffhq/sr8/vsdps/1.0_7.0 --lam 0.8 --xi 7.0
