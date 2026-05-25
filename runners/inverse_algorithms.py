import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
import numpy as np
import torch.optim as optim
from torch.cuda.amp import autocast


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def map_rps(x, seq, model, b, H_funcs, y_0, sigma_0, xi=10.0, optimize_iters=60, vae_lr=0.5, w_prior=2.0, noise_t=10, renoise_t=0, M=1, ps_method='dps', cls_fn=None, classes=None):
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    t = (torch.ones(n) * seq[-1]).to(x.device)
    at = compute_alpha(b, t.long())
    noise = torch.randn_like(x)
    x_T = noise * (1 - at).sqrt()
    xt = x_T
    x0_init = H_funcs.H_pinv(y_0).view(*x.shape)
    x0 = None
    for m in range(M):
        with torch.enable_grad():
            x0_t_with_grad = x0_init.clone().requires_grad_(True)
            optimizer = optim.AdamW([x0_t_with_grad], lr=vae_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_iters, eta_min=1e-5)
            for steps_n in tqdm(range(optimize_iters)):
                optimizer.zero_grad()
                loss_likelihood = torch.sum((y_0-H_funcs.forward(x0_t_with_grad))**2) # MAP
                i = int(noise_t)
                t = (torch.ones(n) * i).to(x.device)
                at = compute_alpha(b, t.long())
                score = None
                N = 1
                for _ in range(N):
                    xt_with_grad = at.sqrt() * x0_t_with_grad + (1-at).sqrt() * torch.randn_like(x0_t_with_grad)
                    et = model(xt_with_grad, t)
                    if et.size(1) == 6:
                        var = torch.exp(et[:, 3:])
                        et = et[:, :3]
                    score = et if score is None else score + et
                x0_t = (xt_with_grad - score / N * (1 - at).sqrt()) / at.sqrt()
                loss_prior = w_prior * torch.sum(score.detach()/N * x0_t_with_grad)
                loss = loss_likelihood + loss_prior
                loss.backward()
                optimizer.step()
                scheduler.step()
        x0_t = x0_t_with_grad.detach()
        xs = [x0_t.to('cpu')]
        x0_preds = [x0_t.to('cpu')]
        xt_next = x0_t
        betas = b.cpu().numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            at_init = torch.tensor(alphas_cumprod[int(renoise_t)-1]) if renoise_t > 0 else torch.tensor(1.0).cuda()
            noise = torch.randn_like(x0_t)
            xt = at_init.sqrt() * x0_t + (1 - at_init).sqrt() * noise
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
                if i >= int(renoise_t):
                    continue
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                with torch.enable_grad():
                    xt_with_grad = xt.clone().requires_grad_(True)
                    if cls_fn == None:
                        et = model(xt_with_grad, t)
                    else:
                        et = model(xt_with_grad, t, classes)
                    # print(et.shape)
                    if et.size(1) == 6:
                        var = torch.exp(et[:, 3:])
                        et = et[:, :3]
                    else:
                        et = et
                        alpha_t_bar = at[0,0,0,0]
                        alpha_t_next_bar = at_next[0,0,0,0]
                        alpha_t = alpha_t_bar/alpha_t_next_bar
                        beta_t = 1-alpha_t
                        sigma_ddpm = ((1-at_next)/(1-at)).sqrt() * (1-at/at_next).sqrt()
                        sigma_tilde = beta_t * (1-at_next) / (1-at)                
                    model_var_values = var
                    min_log = posterior_log_variance_clipped
                    max_log = np.log(betas)
                    min_log = extract_and_expand(min_log, t[0].long(), var)
                    max_log = extract_and_expand(max_log, t[0].long(), var)
                    frac = (model_var_values + 1.0) / 2.0
                    model_log_variance = frac * max_log + (1-frac) * min_log
                    model_variance = torch.exp(model_log_variance * 0.5)
                    x0_t = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
                    x0_t = x0_t.clamp(-1, 1)
                    if ps_method == 'ddnm':
                        eta = 0.8
                        if sigma_0 == 0:
                            x0_t = x0_t + H_funcs.H_pinv(y_0 - H_funcs.H(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
                            add_up = eta * (1-at_next).sqrt() * torch.randn_like(x0_t) + (1-eta**2)**0.5 * (1-at_next).sqrt() * et
                        else:
                            x = torch.randn_like(xt)
                            singulars = H_funcs.singulars()
                            # print(singulars.shape)
                            Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
                            Sigma[:singulars.shape[0]] = singulars
                            Inv_Sigma = 1 / Sigma
                            Inv_Sigma[Sigma==0] = 0
                            U_t_y = H_funcs.Ut(y_0)
                            Sigma = Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
                            Inv_Sigma = Inv_Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
                            V_t_et = H_funcs.Vt(et).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
                            V_t_x0_t = H_funcs.Vt(x0_t).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
            
                            lambda_t = torch.ones_like(V_t_x0_t)
                            sigma_t = (1 - at_next[0,0,0,0]) ** 0.5
                            change_idx = 1.0 * (sigma_t < at_next[0,0,0,0].sqrt()*sigma_0*Inv_Sigma)
                            lambda_t = (1-change_idx) * lambda_t + change_idx * Sigma * sigma_t * (1-eta**2)**0.5/at_next[0,0,0,0].sqrt()/sigma_0
                            random_noise = torch.randn_like(V_t_x0_t)
                            epsilon_tmp = torch.zeros_like(V_t_x0_t)
                            change_idx = 1.0 * (sigma_t >= at_next[0,0,0,0].sqrt()*sigma_0*Inv_Sigma)
                            epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * (sigma_t**2-at_next[0,0,0,0]*sigma_0**2*Inv_Sigma**2) * random_noise
                            change_idx = 1.0 * (sigma_t < at_next[0,0,0,0].sqrt()*sigma_0*Inv_Sigma)
                            epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * eta * sigma_t * random_noise
                            change_idx = 1.0 * (Sigma==0)
                            epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * (sigma_t * (1-eta**2)**0.5 * V_t_et + sigma_t * eta * random_noise)
                            x0_t = x0_t - H_funcs.V((lambda_t * H_funcs.Vt(H_funcs.H_pinv(H_funcs.H(x0_t) - y_0).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])).view(x.shape[0], -1)).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
                            add_up = H_funcs.V(epsilon_tmp.view([epsilon_tmp.shape[0], -1])).view(x.shape)
                        xt_next = at_next.sqrt() * x0_t + add_up
                    else:
                        if ps_method == 'dps':
                            # loss = torch.sum((y_0 - H_funcs.forward(x0_t))**2)
                            loss = torch.linalg.norm(y_0 - H_funcs.forward(x0_t))
                            grad = torch.autograd.grad(loss, [xt_with_grad])[0]
                        elif ps_method == 'pigdm':
                            mat1 = H_funcs.Ut(y_0 - H_funcs.H(x0_t))
                            # mat1 = H_funcs.H_pinv(y) - H_funcs.H_pinv(H_funcs.H(x0_t))
                            mat1 = mat1.view(xt.shape[0], -1).detach()
                            rt = (1-at[0,0,0,0]).sqrt()
                            scale = sigma_0 / rt
                            mat2 = H_funcs.H_scaled_inv(H_funcs.H(x0_t), scale).view(xt.shape[0], -1)
                            loss = torch.sum(mat1 * mat2)
                            grad = torch.autograd.grad(outputs=loss, inputs=xt_with_grad)[0] * at.sqrt()
                        elif ps_method == 'dmps':
                            at_no_bar = at[0,0,0,0]/at_next[0,0,0,0]
                            weight = (1-at_no_bar)/(at_no_bar.sqrt() * at_next)
                            grad = -H_funcs.H_dmps_guidance(xt, y_0, at[0,0,0,0], sigma_0).view(xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]) * weight
                    
                        alpha_t_bar = at[0,0,0,0]
                        alpha_t_next_bar = at_next[0,0,0,0]
                        alpha_t = alpha_t_bar/alpha_t_next_bar
                        beta_t = 1-alpha_t
                        sigma_ddpm = ((1-at_next)/(1-at)).sqrt() * (1-at/at_next).sqrt()
                        sigma_tilde = beta_t * (1-at_next) / (1-at)
                        xt_next = at_next.sqrt() * x0_t + (1-at_next - sigma_ddpm**2).sqrt() * et + model_variance * torch.randn_like(x0_t) - xi * grad
                xt = xt_next.detach()
                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
        if x0 is None:
            x0 = xt_next.to('cpu')
        else:
            x0 += xt_next.to('cpu')
    x0 = x0 / M
    x0_preds = [x0]
    xs = [x0]
    return xs, x0_preds


def lmap_rps(x, seq, model, alphas_cumprod, H_funcs, y_0, sigma_0, lr, N, optimize_iters=200, vae_lr=0.5, w_prior=0.15, noise_t=50, renoise_t=100, lam=0.1, ps_method='dps', stable=False, cls_fn=None, classes=None):
    img = H_funcs.H_pinv(y_0).view([1, 3, 256, 256])
    x0_init = model.encode_first_stage(img)
    pixels_recon = model.decode_first_stage(x0_init)
    n = x.size(0)
    x0_preds = []
    xs = []
    with torch.enable_grad():
        x0_t_with_grad = x0_init.clone().requires_grad_(True)
        optimizer = optim.AdamW([x0_t_with_grad], lr=vae_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_iters, eta_min=1e-5)
        for steps_n in tqdm(range(optimize_iters)):
            optimizer.zero_grad()
            pixels = model.decode_first_stage(x0_t_with_grad)
            loss_likelihood = torch.sum((y_0-H_funcs.forward(pixels))**2) # MAP
            i = int(noise_t)
            # prior
            t = (torch.ones(n) * i).to(x.device)
            at = alphas_cumprod[i]
            # score
            xt_with_grad = at.sqrt() * x0_t_with_grad + (1-at).sqrt() * torch.randn_like(x0_t_with_grad)
            et = model.apply_model(xt_with_grad, t, cond=classes)
            loss_prior = w_prior * torch.sum(et.detach() * x0_t_with_grad)
            loss = loss_likelihood + loss_prior
            loss.backward()
            optimizer.step()
            scheduler.step()
    x0_map = x0_t_with_grad.detach()
    x0_t = x0_map.detach()
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        at_init = alphas_cumprod[int(renoise_t)-1] if renoise_t > 0 else torch.tensor(1.0).cuda()
        noise = torch.randn_like(x0_t)
        xt = at_init.sqrt() * x0_t + (1 - at_init).sqrt() * noise
        xt_map = at_init.sqrt() * x0_map + (1 - at_init).sqrt() * noise
        
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            if i >= int(renoise_t):
                continue
            at = alphas_cumprod[i]
            at_next = alphas_cumprod[j]
            if j < 0:
                at_next = torch.tensor(1.0).cuda()
            # at_next_next = alphas_cumprod[j_next]
            # score
            with torch.enable_grad():
                xt_with_grad = xt.clone().requires_grad_(True)
                t = (torch.ones(n) * i).to(x.device)
                et = model.apply_model(xt_with_grad, t, cond=classes)
                if classes is not None:
                    et1 = et
                    xc_uncond = torch.tensor([1000]) # cfg
                    c_uncond = model.get_learned_conditioning({model.cond_stage_key: xc_uncond.to(model.device)})
                    et2 = model.apply_model(xt_with_grad, t, cond=c_uncond)
                    et_cfg = et1 + 3.0 * (et1 - et2)
                    et = et_cfg
                    x0_t = x0_t_cfg = (xt_with_grad - et_cfg * (1 - at).sqrt()) / at.sqrt()
                else:
                    x0_t = x0_t_cfg = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
                pixels = model.decode_first_stage(x0_t_cfg)
                if ps_method == 'dcdp':
                    x0_t_with_grad = x0_t_cfg.detach().clone().requires_grad_(True)
                    optimizer = optim.AdamW([x0_t_with_grad], lr=lr)
                    for epoch in range(2):
                        optimizer.zero_grad()
                        pixels = model.decode_first_stage(x0_t_with_grad)
                        loss = torch.linalg.norm(y_0 - H_funcs.forward(pixels))
                        loss.backward()
                        optimizer.step()
                    x0_t = x0_t_with_grad.detach()
                    add_up = (1-at_next).sqrt() * torch.randn_like(x0_t)
                    xt_next = at_next.sqrt() * x0_t + add_up
                    xt = xt_next
                elif ps_method == 'ldir':
                    eta = 1.0
                    sigma = eta * ((1-at_next)/(1-at)).sqrt() * (1-at/at_next).sqrt()
                    noise = torch.randn_like(x0_t)
                    xt_next_par = at_next.sqrt() * x0_t_cfg + (1-at_next - sigma**2).sqrt() * et + sigma * noise
                    pixels = model.decode_first_stage(xt_next_par)
                    if stable:
                        loss = (y_0 - H_funcs.forward(pixels)).norm()
                    else:
                        loss = (y_0 - H_funcs.forward(pixels)).norm()**2
                    grad = torch.autograd.grad(outputs=loss, inputs=xt_with_grad)[0]
                    xt_next = xt_next_par - grad * lr * at
                elif ps_method == 'psld':
                    if not stable:
                        loss1 =torch.linalg.norm(y_0 - H_funcs.forward(pixels) - sigma_0 * torch.randn_like(y_0)) ** 2.0
                        pixels_recon = H_funcs.proj(pixels, y_0)
                        loss2 = torch.linalg.norm(x0_t - model.encode_first_stage(pixels_recon.detach())) ** 2.0
                    else:
                        loss1 =torch.linalg.norm(y_0 - H_funcs.forward(pixels) - sigma_0 * torch.randn_like(y_0))
                        pixels_recon = H_funcs.proj(pixels, y_0)
                        loss2 = torch.linalg.norm(x0_t - model.encode_first_stage(pixels_recon.detach()))
                    loss = loss1 + loss2*lam
                    grad = torch.autograd.grad(outputs=loss, inputs=xt_with_grad)[0] * at
                    alpha_t_bar = at
                    alpha_t_next_bar = at_next
                    alpha_t = alpha_t_bar/alpha_t_next_bar
                    beta_t = 1-alpha_t
                    noise = torch.randn_like(x)
                    # DDPM update
                    eta = 1.0
                    sigma = eta * ((1-at_next)/(1-at)).sqrt() * (1-at/at_next).sqrt()
                    noise = torch.randn_like(x0_t)
                    xt_next = at_next.sqrt() * x0_t_cfg + (1-at_next - sigma**2).sqrt() * et + sigma * noise - lr * grad
                else:
                    # DPS
                    if not stable:
                        loss = torch.linalg.norm(y_0 - H_funcs.forward(pixels)) ** 2.0
                    else:
                        loss = torch.linalg.norm(y_0 - H_funcs.forward(pixels))
                    
                    grad = torch.autograd.grad(outputs=loss, inputs=xt_with_grad)[0] * at
                    alpha_t_bar = at
                    alpha_t_next_bar = at_next
                    alpha_t = alpha_t_bar/alpha_t_next_bar
                    beta_t = 1-alpha_t
                    noise = torch.randn_like(x)
                    # DDPM update
                    eta = 1.0
                    sigma = eta * ((1-at_next)/(1-at)).sqrt() * (1-at/at_next).sqrt()
                    noise = torch.randn_like(x0_t)
                    xt_next = at_next.sqrt() * x0_t_cfg + (1-at_next - sigma**2).sqrt() * et + sigma * noise - lr * grad
            xt = xt_next.detach()
    with torch.no_grad():
        img = model.decode_first_stage(xt)
    x0_preds.append(img.to('cpu'))
    xs.append(img.to('cpu'))
    return xs, x0_preds


def lmap_rps_t2i(x, seq, model, alphas_cumprod, H_funcs, y_0, sigma_0, lr, N, optimize_iters=200, vae_lr=0.5, w_prior=0.15, noise_t=50, renoise_t=100, lam=0.1, eta_min=1e-5, ps_method='latent_dps', stable=False, prompt=None, classes=None):
    img = H_funcs.H_pinv(y_0).view([1, 3, 512, 512])
    x0_init = model.encode_first_stage(img)
    pixels_recon = model.decode_first_stage(x0_init)
    dis = torch.mean((pixels_recon - img)**2).item()
    n = x.size(0)
    x0_preds = []
    xs = []
    w_prior_init = w_prior
    with torch.enable_grad():
        x0_t_with_grad = x0_init.clone().requires_grad_(True)
        optimizer = optim.AdamW([x0_t_with_grad], lr=vae_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_iters, eta_min=eta_min)
        for steps_n in tqdm(range(optimize_iters)):
            optimizer.zero_grad()
            pixels = model.decode_first_stage(x0_t_with_grad)
            if not stable:
                loss_likelihood = torch.sum((y_0-H_funcs.forward(pixels))**2) # MAP
            else:
                loss_likelihood = torch.linalg.norm(y_0-H_funcs.forward(pixels))
            i = int(noise_t)
            # prior
            t = (torch.ones(n) * i).to(x.device)
            at = alphas_cumprod[i]
            # score
            xt_with_grad = at.sqrt() * x0_t_with_grad + (1-at).sqrt() * torch.randn_like(x0_t_with_grad)
            et = model.apply_model(xt_with_grad, t, prompt=prompt)
            x0_t = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
            loss_prior = w_prior * torch.sum(et.detach() * x0_t_with_grad)
            loss = loss_likelihood + loss_prior
            loss.backward()
            optimizer.step()
            scheduler.step()
    x0_map = x0_t_with_grad.detach()
    x0_t = x0_t_with_grad.detach()
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        at_init = alphas_cumprod[int(renoise_t)-1] if renoise_t > 0 else torch.tensor(1.0).cuda()
        noise = torch.randn_like(x0_t)
        xt = at_init.sqrt() * x0_t + (1 - at_init).sqrt() * noise
        xt_map = at_init.sqrt() * x0_map + (1 - at_init).sqrt() * noise
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            if i >= int(renoise_t):
                continue
            at = alphas_cumprod[i]
            at_next = alphas_cumprod[j]
            if j < 0:
                at_next = torch.tensor(1.0).cuda()
            if ps_method == 'latent_dcdp':
                t = (torch.ones(n) * i).to(x.device)
                et = model.apply_model(xt, t, prompt=prompt)
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                with torch.enable_grad():
                    x0_t_with_grad = x0_t.clone().requires_grad_(True)
                    optimizer = optim.AdamW([x0_t_with_grad], lr=0.1)
                    for epoch in range(100):
                        optimizer.zero_grad()
                        pixels = model.decode_first_stage(x0_t_with_grad)
                        loss = torch.sum((y_0 - H_funcs.forward(pixels))**2)
                        loss.backward()
                        optimizer.step()
                x0_t = x0_t_with_grad.detach()
                add_up = (1-at_next).sqrt() * torch.randn_like(x0_t)
                xt_next = at_next.sqrt() * x0_t + add_up
                xt = xt_next
            else:
                with torch.enable_grad():
                    xt_with_grad = xt.clone().requires_grad_(True)
                    t = (torch.ones(n) * i).to(x.device)
                    et = model.apply_model(xt_with_grad, t, prompt=prompt)
                    et1 = et
                    et2 = (xt_with_grad - at.sqrt() * x0_map) / (1 - at).sqrt()
                    x0_t = (xt_with_grad - et1 * (1 - at).sqrt()) / at.sqrt()
                    et_cfg = et1 + 0.0 * (et1 - et2)
                    x0_t_cfg = (xt_with_grad - et_cfg * (1 - at).sqrt()) / at.sqrt()
                    pixels = model.decode_first_stage(x0_t_cfg)
                    if ps_method == 'psld':
                        loss1 =torch.linalg.norm(y_0 - H_funcs.forward(pixels) - sigma_0 * torch.randn_like(y_0)) ** 2.0
                        pixels_recon = H_funcs.proj(pixels, y_0)
                        loss2 = torch.linalg.norm(x0_t - model.encode_first_stage(pixels_recon.detach())) ** 2.0
                        loss = loss1 + loss2*lam
                        grad = torch.autograd.grad(outputs=loss, inputs=xt_with_grad)[0] * at
                    else:
                        # DPS
                        pixels = pixels.clamp(-1, 1)
                        if not stable:
                            loss = torch.linalg.norm(y_0 - H_funcs.forward(pixels)) ** 2.0
                        else:
                            loss = torch.linalg.norm(y_0 - H_funcs.forward(pixels))
                        grad = torch.autograd.grad(outputs=loss, inputs=xt_with_grad)[0] * at
                alpha_t_bar = at
                alpha_t_next_bar = at_next
                alpha_t = alpha_t_bar/alpha_t_next_bar
                beta_t = 1-alpha_t
                noise = torch.randn_like(x)
                eta = 1.0
                sigma = eta * ((1-at_next)/(1-at)).sqrt() * (1-at/at_next).sqrt()
                noise = torch.randn_like(x0_t)
                xt_next = at_next.sqrt() * x0_t_cfg + (1-at_next - sigma**2).sqrt() * et + sigma * noise - lr * grad
                xt_map_next = beta_t * at_next.sqrt() / (1-at) * x0_map + (1 - at_next) / (1 - at) * alpha_t.sqrt() * xt + sigma * noise - lr * grad
                xt_map = xt_map_next
                xt = xt_next
    img = model.decode_first_stage(xt)
    x0_preds.append(img.to('cpu'))
    xs.append(img.to('cpu'))
    return xs, x0_preds