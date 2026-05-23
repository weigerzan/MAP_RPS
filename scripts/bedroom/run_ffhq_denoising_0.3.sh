# 全部重新调
CUDA_VISIBLE_DEVICES=6 python main.py --ni --config bedroom.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i bedroom/denoise_0.3/vsdps/1.0 --lam 1.0 --xi 2.0 # 0.1915 / 0.03614
CUDA_VISIBLE_DEVICES=4 python main.py --ni --config bedroom.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i bedroom/denoise_0.3/vsdps/0.8 --lam 0.8 --xi 8.0 # 0.2638 / 0.03559
CUDA_VISIBLE_DEVICES=3 python main.py --ni --config bedroom.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i bedroom/denoise_0.3/vsdps/0.5 --lam 0.5 --xi 4.0 # 0.2900 / 0.03456
CUDA_VISIBLE_DEVICES=3 python main.py --ni --config bedroom.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i bedroom/denoise_0.3/vsdps/0.3 --lam 0.3 --xi 8.0 # 0.3033 / 0.03453
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config bedroom.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i bedroom/denoise_0.3/vsdps/0.0 --lam 0.0 --xi 6.0 # 0.3333 / 0.03408

