CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 900 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/900 --lam 1.0 --xi 6.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 800 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/800 --lam 1.0 --xi 7.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 700 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/700 --lam 1.0 --xi 7.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 600 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/600 --lam 1.0 --xi 8.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 500 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/500 --lam 1.0 --xi 10.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 400 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/400 --lam 1.0 --xi 9.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 300 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/300 --lam 1.0 --xi 11.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 200 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/200 --lam 1.0 --xi 15.0
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config ffhq.yml --timesteps 100 --deg deblur_aniso --sigma_0 0.0 -i ffhq/deblur_aniso/dps_s/100 --lam 1.0 --xi 18.0

