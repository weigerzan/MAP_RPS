CUDA_VISIBLE_DEVICES=1 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_a/2 --lam 1.0 --xi 2.0 --M 2
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_a/4 --lam 1.0 --xi 2.0 --M 4
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_a/8 --lam 1.0 --xi 2.0 --M 8
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_a/16 --lam 1.0 --xi 2.0 --M 16
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_a/32 --lam 1.0 --xi 2.0 --M 32
\

CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg deblur_aniso --sigma_0 0.0 -i ffhq/test_time --lam 1.0 --xi 2.0 --M 2
