CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/dps_a/2 --lam 1.0 --xi 1.0 --M 2
CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/dps_a/4 --lam 1.0 --xi 1.0 --M 4
CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/dps_a/8 --lam 1.0 --xi 1.0 --M 8
CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/dps_a/16 --lam 1.0 --xi 1.0 --M 16
CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/inp_0.1/dps_a/32 --lam 1.0 --xi 1.0 --M 32


CUDA_VISIBLE_DEVICES=5 python main.py --ni --config ffhq.yml --timesteps 1000 --deg inp --sigma_0 0.1 -i ffhq/test_time --lam 1.0 --xi 1.0 --M 2

