# inpainting
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg inp --sigma_0 0.05 -i coco/inp_0.05/lmap_ps/0 --algo lmap_rps_t2i --lr 0.01 --vae_lr 2.0 --optimize_iters 200 --w_prior 0.3 --noise_t 50 --renoise_t 0 --lam 0.1 --eta_min 1e-2
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg inp --sigma_0 0.05 -i coco/inp_0.05/lmap_ps/600 --algo lmap_rps_t2i --lr 0.01 --vae_lr 2.0 --optimize_iters 200 --w_prior 0.3 --noise_t 50 --renoise_t 600 --lam 0.1 --eta_min 1e-2

# super-resolution
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg sr4 --sigma_0 0.05 -i coco/sr4_0.05/lmap_ps/0 --algo lmap_rps_t2i --lr 0.1 --vae_lr 2.0 --optimize_iters 100 --w_prior 0.15 --noise_t 50 --renoise_t 0 --lam 0.1 --eta_min 1e-2
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg sr4 --sigma_0 0.05 -i coco/sr4_0.05/lmap_ps/600 --algo lmap_rps_t2i --lr 0.1 --vae_lr 2.0 --optimize_iters 100 --w_prior 0.15 --noise_t 50 --renoise_t 600 --lam 0.1 --eta_min 1e-2

# anisotropic deblurring
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg deblur_aniso --sigma_0 0.05 -i coco/deblur_aniso_0.05/lmap_ps/0 --algo lmap_rps_t2i --lr 0.1 --vae_lr 2.0 --optimize_iters 200 --w_prior 0.1 --noise_t 50 --renoise_t 0 --lam 0.1  --eta_min 1e-2 --ps_method dcdp
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg deblur_aniso --sigma_0 0.05 -i coco/deblur_aniso_0.05/lmap_ps/0 --algo lmap_rps_t2i --lr 0.1 --vae_lr 2.0 --optimize_iters 200 --w_prior 0.1 --noise_t 50 --renoise_t 0 --lam 0.1  --eta_min 1e-2 --ps_method dcdp

# compressed sensing
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg cs2 --sigma_0 0.05 -i coco/cs2_0.05/lmap_ps/0 --algo lmap_rps_t2i --lr 0.1 --vae_lr 0.5 --optimize_iters 300 --w_prior 0.005 --noise_t 50 --renoise_t 0 --lam 0.1 --stable --eta_min 1e-2
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg cs2 --sigma_0 0.05 -i coco/cs2_0.05/lmap_ps/600 --algo lmap_rps_t2i --lr 1.5 --vae_lr 0.5 --optimize_iters 300 --w_prior 0.005 --noise_t 50 --renoise_t 0 --lam 0.1 --stable --eta_min 1e-2

# hdr
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg hdr --sigma_0 0.05 -i coco/hdr_0.05/lmap_ps/0 --algo lmap_rps_t2i --lr 0.1 --vae_lr 2.0 --optimize_iters 400 --w_prior 0.005 --noise_t 50 --renoise_t 0 --lam 0.1 --stable --eta_min 1e-2
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg hdr --sigma_0 0.05 -i coco/hdr_0.05/lmap_ps/600 --algo lmap_rps_t2i --lr 0.5 --vae_lr 2.0 --optimize_iters 400 --w_prior 0.005 --noise_t 50 --renoise_t 600 --lam 0.1 --stable --eta_min 1e-2

# nonlinear deblurring
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg deblur_nonlinear --sigma_0 0.05 -i coco/deblur_nonlinear_0.05/lmap_ps/0 --algo lmap_rps_t2i --lr 0.1 --vae_lr 2.0 --optimize_iters 500 --w_prior 0.002 --noise_t 50 --renoise_t 0 --lam 0.1 --stable --eta_min 1e-5
python main.py --ni --config sd15/sd15.yaml --timesteps 1000 --deg deblur_nonlinear --sigma_0 0.05 -i coco/deblur_nonlinear_0.05/lmap_ps/600 --algo lmap_rps_t2i --lr 1.5 --vae_lr 2.0 --optimize_iters 500 --w_prior 0.002 --noise_t 50 --renoise_t 0 --lam 0.1 --stable --eta_min 1e-5