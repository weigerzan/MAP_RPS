# 全部重新调
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/vsdps/1.0_chazhi --lam 1.0 --xi 2.0
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/vsdps/0.8_chazhi --lam 0.8 --xi 6.0
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/vsdps/0.5_chazhi --lam 0.5 --xi 12.0
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/vsdps/0.3 --lam 0.3 --xi 14.0
CUDA_VISIBLE_DEVICES=7 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/vsdps/0.0 --lam 0.0 --xi 17.0


