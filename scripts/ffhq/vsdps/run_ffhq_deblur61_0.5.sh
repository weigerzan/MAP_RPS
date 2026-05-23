# 全部重新调
CUDA_VISIBLE_DEVICES=3 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.5 -i ffhq/deblur_gaussian_61_0.5/vsdps/1.0 --lam 1.0 --xi 2.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.5 -i ffhq/deblur_gaussian_61_0.5/vsdps/0.8 --lam 0.8 --xi 5.0
CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.5 -i ffhq/deblur_gaussian_61_0.5/vsdps/0.5 --lam 0.5 --xi 8.0
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.5 -i ffhq/deblur_gaussian_61_0.5/vsdps/0.3 --lam 0.3 --xi 12.0
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_gaussian_61 --sigma_0 0.5 -i ffhq/deblur_gaussian_61_0.5/vsdps/0.0 --lam 0.0 --xi 16.0


