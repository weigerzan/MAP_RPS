python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_rps/0 --algo map_rps --lr 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 0
python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_rps/test --algo map_rps --lr 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 25
python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_rps/50 --algo map_rps --lr 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 50
python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_rps/75 --algo map_rps --lr 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 75
python main.py --ni --config ffhq.yml --timesteps 1000 --deg denoise --sigma_0 0.3 -i ffhq/denoise_0.3/map_rps/100 --algo map_rps --lr 1.0 --optimize_iters 60 --vae_lr 0.5 --w_prior 2.0 --noise_t 10 --renoise_t 100
 

