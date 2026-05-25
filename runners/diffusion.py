import os
import logging
import time
import glob

from skimage.metrics import structural_similarity as ssim
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from datasets import get_dataset, data_transform, inverse_data_transform
from runners.inverse_algorithms import map_rps, lmap_rps, lmap_rps_t2i
import lpips

import torchvision.utils as tvu
import random
import yaml
from PIL import Image
from torchvision import transforms


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        if hasattr(config.model, 'var_type') and config.model.var_type is not None:
            self.model_var_type = config.model.var_type
            betas = get_beta_schedule(
                beta_schedule=config.diffusion.beta_schedule,
                beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
            )
            betas = self.betas = torch.from_numpy(betas).float().to(self.device)
            self.num_timesteps = betas.shape[0]

            alphas = 1.0 - betas
            alphas_cumprod = alphas.cumprod(dim=0)
            alphas_cumprod_prev = torch.cat(
                [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
            )
            self.alphas_cumprod_prev = alphas_cumprod_prev
            posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )
            if self.model_var_type == "fixedlarge":
                self.logvar = betas.log()
            elif self.model_var_type == "fixedsmall":
                self.logvar = posterior_variance.clamp(min=1e-20).log()
        else:
            print("Use alphas from model config.")

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'ffhq_pixel':
            cls_fn = None
            from third_party.guided_diffusion.unet_ffhq import create_model as create_model_ffhq
            model_config = load_yaml('configs/ffhq_model_config.yaml')
            model = create_model_ffhq(**model_config)
            model = model.to(self.device)
            model.eval()
        elif self.config.model.type == 'ffhq_ldm':
            from third_party.ldm.utils_ldm import get_model
            model, config = get_model('pretrained/ffhq_ldm/ldm/ffhq256/model.ckpt')
            self.alphas_cumprod = model.alphas_cumprod.cuda()
            self.num_timesteps = len(self.alphas_cumprod)
            model.eval()
            model.model.diffusion_model.eval()
            model.first_stage_model.eval()
            model = model.cuda()
            cls_fn = None
        elif self.config.model.type == 'sd15':
            from third_party.sd15.sd15 import SD15
            model = SD15(model_id=self.config.model.model_id, device=self.device)
            self.alphas_cumprod = model.alphas_cumprod.cuda()
            self.num_timesteps = len(self.alphas_cumprod)
        else:
            raise NotImplementedError(
                    f"Model type {self.config.model.type} not implemented"
                )
        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)        
        device_count = torch.cuda.device_count()
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)
        print(f'Dataset has size {len(test_dataset)}')    
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=g,
        )
        

        ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        
        if 'sr' in deg:
            if deg[:10] == 'sr_bicubic':
                factor = int(deg[10:])
                from obs_functions.Hfuncs import SRConv
                def bicubic_kernel(x, a=-0.5):
                    if abs(x) <= 1:
                        return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                    elif 1 < abs(x) and abs(x) < 2:
                        return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                    else:
                        return 0
                k = np.zeros((factor * 4))
                for i in range(factor * 4):
                    x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                    k[i] = bicubic_kernel(x)
                k = k / np.sum(k)
                kernel = torch.from_numpy(k).float().to(self.device)
                H_funcs = SRConv(kernel / kernel.sum(), \
                                self.config.data.channels, self.config.data.image_size, self.device, stride = factor)
            else:
                # Super-Resolution
                blur_by = int(deg[2:])
                from obs_functions.Hfuncs import SuperResolution
                H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif 'inp' in deg:
            # Random inpainting
            missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(self.device).long()
            from obs_functions.Hfuncs import Inpainting
            H_funcs = Inpainting(config.data.channels, config.data.image_size, missing_r, self.device)
        elif 'cs' in deg:
            compress_by = int(deg[2:])
            from obs_functions.Hfuncs import WalshHadamardCS
            H_funcs = WalshHadamardCS(self.config.data.channels, self.config.data.image_size, compress_by, torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        elif deg == 'deblur_aniso':
            from obs_functions.Hfuncs import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), self.config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'denoise':
            from obs_functions.Hfuncs import Denoising
            H_funcs = Denoising(config.data.channels, config.data.image_size, self.device)
        elif deg == 'deblur_gaussian_61':
            from obs_functions.Hfuncs import GaussianBlurOperator
            H_funcs = GaussianBlurOperator(kernel_size=61, intensity=3.0, device=self.device)
        elif 'deblur_gauss' in deg:
            # Gaussian Deblurring
            from obs_functions.Hfuncs import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif 'phase' in deg:
            # Phase Retrieval
            from obs_functions.Hfuncs import PhaseRetrievalOperator
            H_funcs = PhaseRetrievalOperator(oversample=2.0, device=self.device)
        elif 'hdr' in deg:
            # HDR
            from obs_functions.Hfuncs import HDR
            H_funcs = HDR()  
        elif deg == 'deblur_nonlinear':
            from obs_functions.Hfuncs import NonlinearBlurOperator
            H_funcs = NonlinearBlurOperator(self.device, opt_yml_path='./bkse/options/generate_blur/default.yml')   
        else:
            print("ERROR: degradation type not supported")
            quit()

        # for linear observations
        if 'sr' in deg or 'inp' in deg or 'deblur_gauss' in deg:
            args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0
        lr = args.lr
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        avg_rmse = 0.0
        pbar = tqdm.tqdm(val_loader)
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        classes = None
        prompt = None
        with torch.no_grad():
            for batch in pbar:
                if self.config.data.dataset == 'coco':
                    x_orig, prompt = batch['image'], batch['caption']
                else:
                    x_orig, classes = batch
                classes = None if 'ffhq' in self.config.model.type else classes
                x_orig = x_orig.to(self.device)
                x_orig = data_transform(self.config, x_orig)

                y_0 = H_funcs.forward(x_orig)
                y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
                y_pinv = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
                # print(y_0.shape)
                for i in range(len(y_0)):
                    tvu.save_image(
                        inverse_data_transform(config, y_pinv[i]), os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, x_orig[i]), os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
                    )
                ##Begin DDIM
                x = torch.randn(
                    y_0.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                with torch.no_grad():
                    x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, args.lam, args.lr, args.M, last=False, cls_fn=cls_fn, classes=classes, prompt=prompt)
                x = [inverse_data_transform(config, y) for y in x]
                for i in [-1]: #range(len(x)):
                    for j in range(x[i].size(0)):
                        tvu.save_image(
                            x[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                        )
                        if i == len(x)-1 or i == -1:
                            orig = inverse_data_transform(config, x_orig[j])
                            # print(torch.norm(orig[0]))
                            mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                            psnr = 10 * torch.log10(1 / mse)
                            avg_psnr += psnr
                            avg_rmse += mse.sqrt()
                            # print(x[i][j].shape)
                            avg_ssim += ssim(x[i][j].numpy(), orig.cpu().numpy(), data_range=x[i][j].numpy().max() - x[i][j].numpy().min(), channel_axis=0)
                            LPIPS = loss_fn_vgg(2*orig-1.0, 2*torch.tensor(x[i][j]).to(torch.float32).cuda()-1.0)
                            avg_lpips += LPIPS[0,0,0,0]
                idx_so_far += y_0.shape[0]

                print("PSNR:{}, SSIM:{}, LPIPS:{}, RMSE:{}".format(avg_psnr / (idx_so_far - idx_init), avg_ssim / (idx_so_far - idx_init), avg_lpips / (idx_so_far - idx_init), avg_rmse / (idx_so_far - idx_init)))

            avg_psnr = avg_psnr / (idx_so_far - idx_init)
            print("Total Average PSNR: %.2f" % avg_psnr)
            print("Number of samples: %d" % (idx_so_far - idx_init))

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, lam, lr, M=1, last=True, cls_fn=None, classes=None, prompt=None):
        skip = self.num_timesteps / self.args.timesteps
        seq = []
        for k in range(self.args.timesteps):
            seq.append(int(k*skip))
        if self.args.algo == 'map_rps':
            x = map_rps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, lr, M=M, optimize_iters=self.args.optimize_iters, vae_lr=self.args.vae_lr, w_prior=self.args.w_prior, noise_t=self.args.noise_t, renoise_t=self.args.renoise_t, ps_method=self.args.ps_method, cls_fn=cls_fn, classes=classes)
        elif self.args.algo == 'lmap_rps':
            x = lmap_rps(x, seq, model, self.alphas_cumprod, H_funcs, y_0, sigma_0, self.args.lr, self.args.M, optimize_iters=self.args.optimize_iters, vae_lr=self.args.vae_lr, w_prior=self.args.w_prior, noise_t=self.args.noise_t, renoise_t=self.args.renoise_t, lam=lam, ps_method=self.args.ps_method, stable=self.args.stable, cls_fn=cls_fn, classes=classes)
        elif self.args.algo == 'lmap_rps_t2i':
            x = lmap_rps_t2i(x, seq, model, self.alphas_cumprod, H_funcs, y_0, sigma_0, self.args.lr, self.args.M, optimize_iters=self.args.optimize_iters, vae_lr=self.args.vae_lr, w_prior=self.args.w_prior, noise_t=self.args.noise_t, renoise_t=self.args.renoise_t, lam=lam, ps_method=self.args.ps_method, eta_min=self.args.eta_min, stable=self.args.stable, prompt=prompt, classes=classes)
        else:
            raise NotImplementedError(f"Algorithm {self.args.algo} not implemented")
        if last:
            x = x[0][-1]
        return [x[0][-1]], None