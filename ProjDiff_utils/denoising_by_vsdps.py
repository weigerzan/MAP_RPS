import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
import numpy as np
import torch.optim as optim


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, lam=1.0, xi=10.0, M=1, cls_fn=None, classes=None):
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    #setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    t = (torch.ones(n) * seq[-1]).to(x.device)
    at = compute_alpha(b, t.long())
    noise = torch.randn_like(x)
    x_T = noise * (1 - at).sqrt()
    xt = x_T
    
    betas = b.cpu().numpy()
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # log calculation clipped because the posterior variance is 0 at the
    # beginning of the diffusion chain.
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    )
    x0 = None
    for m in range(M):
        with torch.no_grad():
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
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
                    # print(var)
                    # 计算var                   
                    model_var_values = var
                    min_log = posterior_log_variance_clipped
                    max_log = np.log(betas)
                    # print(t[0])
                    min_log = extract_and_expand(min_log, t[0].long(), var)
                    max_log = extract_and_expand(max_log, t[0].long(), var)

                    # The model_var_values is [-1, 1] for [min_var, max_var]
                    frac = (model_var_values + 1.0) / 2.0
                    model_log_variance = frac * max_log + (1-frac) * min_log
                    model_variance = torch.exp(model_log_variance * 0.5)
                    # print(model_variance)
                    x0_t = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
                    x0_t_clamp = x0_t.clamp(-1, 1)
                    # print(x0_t)
                    loss = torch.linalg.norm(y_0 - H_funcs.forward(x0_t_clamp))
                    grad = torch.autograd.grad(loss, [xt_with_grad])[0]
                alpha_t_bar = at[0,0,0,0]
                alpha_t_next_bar = at_next[0,0,0,0]
                alpha_t = alpha_t_bar/alpha_t_next_bar
                beta_t = 1-alpha_t
                sigma_ddpm = ((1-at_next)/(1-at)).sqrt() * (1-at/at_next).sqrt()
                sigma_tilde = beta_t * (1-at_next) / (1-at)
                # xt_next = at_next.sqrt() * x0_t + (1-at_next - sigma_ddpm**2).sqrt() * et + lam * model_variance * torch.randn_like(x0_t) - xi * grad
                xt = (at.sqrt() * x0_t_clamp + (1-at).sqrt() * et) * (at**0.5) + xt * (1-at**0.5) # 增加稳定性
                xt_next = beta_t * at_next.sqrt() / (1-at) * x0_t_clamp + (1 - at_next) / (1 - at) * alpha_t.sqrt() * xt + lam * model_variance * torch.randn_like(x0_t_clamp) - xi * grad
                xt = xt_next
                # x0_preds.append(x0_t.to('cpu'))
                # xs.append(xt_next.to('cpu'))
        if x0 is None:
            x0 = xt.to('cpu')
        else:
            x0 += xt.to('cpu')
    x0 = x0 / M
    x0_preds = [x0]
    xs = [x0]
    return xs, x0_preds


def reddiff_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, lam=1.0, xi=10.0, M=1, cls_fn=None, classes=None):
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    #setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    t = (torch.ones(n) * seq[-1]).to(x.device)
    at = compute_alpha(b, t.long())
    noise = torch.randn_like(x)
    x_T = noise * (1 - at).sqrt()
    xt = x_T
    
    betas = b.cpu().numpy()
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # log calculation clipped because the posterior variance is 0 at the
    # beginning of the diffusion chain.
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    )
    x0 = None
    for m in range(M):
        with torch.no_grad():
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                # with torch.enable_grad():
                xt_with_grad = xt.clone()
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
                # print(var)
                # 计算var                   
                model_var_values = var
                min_log = posterior_log_variance_clipped
                max_log = np.log(betas)
                # print(t[0])
                min_log = extract_and_expand(min_log, t[0].long(), var)
                max_log = extract_and_expand(max_log, t[0].long(), var)

                # The model_var_values is [-1, 1] for [min_var, max_var]
                frac = (model_var_values + 1.0) / 2.0
                model_log_variance = frac * max_log + (1-frac) * min_log
                model_variance = torch.exp(model_log_variance * 0.5)
                # print(model_variance)
                x0_t = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
                x0_t_clamp = x0_t.clamp(-1, 1)
                # print(x0_t)
                with torch.enable_grad():
                    if x0 is None:
                        x0 = x0_t_clamp.clone()
                    x0_with_grad = x0_t_clamp.clone().requires_grad_(True)
                    loss = torch.linalg.norm(y_0 - H_funcs.forward(x0_with_grad))
                    grad = torch.autograd.grad(loss, [x0_with_grad])[0]
                x0_next = x0 + xi *((x0_with_grad - x0) * lam - grad)
                x0 = x0_next.detach().clip(-1, 1)
                xt = at_next.sqrt() * x0 + (1-at_next).sqrt() * torch.randn_like(x0)                   
                # x0_preds.append(x0_t.to('cpu'))
                # xs.append(xt_next.to('cpu'))
        # if x0 is None:
        #     x0 = xt.to('cpu')
        # else:
        #     x0 += xt.to('cpu')
    # x0 = x0 / M
    x0_preds = [x0.detach().cpu()]
    xs = [x0.detach().cpu()]
    return xs, x0_preds

# TODO
def dmap_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, lam=1.0, xi=10.0, M=1, cls_fn=None, classes=None):
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    #setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    t = (torch.ones(n) * seq[-1]).to(x.device)
    at = compute_alpha(b, t.long())
    noise = torch.randn_like(x)
    x_T = noise * (1 - at).sqrt()
    xt = x_T
    
    betas = b.cpu().numpy()
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # log calculation clipped because the posterior variance is 0 at the
    # beginning of the diffusion chain.
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    )
    x0 = None
    N = 2
    d = x.size(1) * x.size(2) * x.size(3)
    with torch.no_grad():
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            # with torch.enable_grad():
            xt_with_grad = xt.clone()
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
                # print(var)
            model_var_values = var
            min_log = posterior_log_variance_clipped
            max_log = np.log(betas)
            # print(t[0])
            min_log = extract_and_expand(min_log, t[0].long(), var)
            max_log = extract_and_expand(max_log, t[0].long(), var)

            # The model_var_values is [-1, 1] for [min_var, max_var]
            frac = (model_var_values + 1.0) / 2.0
            model_log_variance = frac * max_log + (1-frac) * min_log
            model_variance = torch.exp(model_log_variance * 0.5)
            # print(model_variance)
            x0_t = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
            x0_t_clamp = x0_t.clamp(-1, 1)
            # print(x0_t)
            # loss = torch.linalg.norm(y_0 - H_funcs.forward(x0_t_clamp))
            # grad = torch.autograd.grad(loss, [xt_with_grad])[0]
            alpha_t_bar = at[0,0,0,0]
            alpha_t_next_bar = at_next[0,0,0,0]
            alpha_t = alpha_t_bar/alpha_t_next_bar
            beta_t = 1-alpha_t
            sigma_ddpm = ((1-at_next)/(1-at)).sqrt() * (1-at/at_next).sqrt()
            sigma_tilde = beta_t * (1-at_next) / (1-at)
            # xt_next = at_next.sqrt() * x0_t + (1-at_next - sigma_ddpm**2).sqrt() * et + lam * model_variance * torch.randn_like(x0_t) - xi * grad
            xt = (at.sqrt() * x0_t_clamp + (1-at).sqrt() * et) * (at**0.5) + xt * (1-at**0.5) # 增加稳定性
            xt_next = beta_t * at_next.sqrt() / (1-at) * x0_t_clamp + (1 - at_next) / (1 - at) * alpha_t.sqrt() * xt + lam * model_variance * torch.randn_like(x0_t_clamp)
            # dmap update
            mu_t = beta_t * at_next.sqrt() / (1-at) * x0_t_clamp + (1 - at_next) / (1 - at) * alpha_t.sqrt() * xt
            if j >= 0:
                for _ in range(N):
                    with torch.enable_grad():
                        xt_with_grad = xt_next.clone().requires_grad_(True)
                        et = model(xt_with_grad, t)[:, :3]
                        x0_t = (xt_with_grad - et * (1 - at_next).sqrt()) / at_next.sqrt()
                        loss = torch.norm(y_0 - H_funcs.forward(x0_t))
                        grad = torch.autograd.grad(outputs=loss, inputs=xt_with_grad)[0]
                        # grad = grad
                    xt_next = xt_next - xi * grad * at_next
                    xt_next = mu_t + (xt_next - mu_t) / torch.norm(xt_next - mu_t) * lam * model_variance * (d ** 0.5)
            
            xt = xt_next
            # x0_preds.append(x0_t.to('cpu'))
            # xs.append(xt_next.to('cpu'))
    x0 = xt.detach().cpu()
    x0_preds = [x0]
    xs = [x0]
    return xs, x0_preds


def ccdf_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, xi=10.0, optimize_iters=60, vae_lr=0.5, w_prior=2.0, noise_t=10, renoise_t=0, M=1, ps_method='dps', cls_fn=None, classes=None):
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    #setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    t = (torch.ones(n) * seq[-1]).to(x.device)
    at = compute_alpha(b, t.long())
    noise = torch.randn_like(x)
    x_T = noise * (1 - at).sqrt()
    xt = x_T
    
    # 先做MAP
    x0_init = H_funcs.H_pinv(y_0).view(*x.shape)
    # i = int(50)
    # # prior
    # t = (torch.ones(n) * i).to(x.device)
    # # at = alphas_cumprod[i]
    # at = compute_alpha(b, t.long())
    # # score
    # xt_with_grad = at.sqrt() * x0_init + (1-at).sqrt() * torch.randn_like(x0_init)
    # et = model(xt_with_grad, t)
    # if et.size(1) == 6:
    #     var = torch.exp(et[:, 3:])
    #     et = et[:, :3]
    # x0_init = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
    # x0_init = torch.randn_like(x)
    # x0_init = x0_init.clamp(-1, 1)
    x0 = None
    for m in range(M):
        x0_t = x0_init.detach()
        xs = [x0_t.to('cpu')]
        x0_preds = [x0_t.to('cpu')]
        # print(xs)
        # print(x0_preds)
        # print(11111111111111111111)
        xt_next = x0_t
        # 接一个DPS
        betas = b.cpu().numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )

        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            # seq_next_next = [-1, -1] + list(seq[:-2])
            # t = (torch.ones(n) * int(renoise_t)).to(x.device)
            # if renoise_t > 0:
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
                    # print(var)
                    # 计算var                   
                    model_var_values = var
                    min_log = posterior_log_variance_clipped
                    max_log = np.log(betas)
                    # print(t[0])
                    min_log = extract_and_expand(min_log, t[0].long(), var)
                    max_log = extract_and_expand(max_log, t[0].long(), var)

                    # The model_var_values is [-1, 1] for [min_var, max_var]
                    frac = (model_var_values + 1.0) / 2.0
                    model_log_variance = frac * max_log + (1-frac) * min_log
                    model_variance = torch.exp(model_log_variance * 0.5)
                    # print(model_variance)
                    x0_t = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
                    x0_t = x0_t.clamp(-1, 1)
                    # print(x0_t)
                    # loss = torch.linalg.norm(y_0 - H_funcs.forward(x0_t))
                    if ps_method == 'ddnm':
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
                # xt_next = beta_t * at_next.sqrt() / (1-at) * x0_t + (1 - at_next) / (1 - at) * alpha_t.sqrt() * xt + 1.0 * model_variance * torch.randn_like(x0_t) - xi * grad
                xt = xt_next
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

def map_ps_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, xi=10.0, optimize_iters=60, vae_lr=0.5, w_prior=2.0, noise_t=10, renoise_t=0, M=1, ps_method='dps', cls_fn=None, classes=None):
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    #setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    t = (torch.ones(n) * seq[-1]).to(x.device)
    at = compute_alpha(b, t.long())
    noise = torch.randn_like(x)
    x_T = noise * (1 - at).sqrt()
    xt = x_T
    
    # 先做MAP
    x0_init = H_funcs.H_pinv(y_0).view(*x.shape)
    # i = int(50)
    # # prior
    # t = (torch.ones(n) * i).to(x.device)
    # # at = alphas_cumprod[i]
    # at = compute_alpha(b, t.long())
    # # score
    # xt_with_grad = at.sqrt() * x0_init + (1-at).sqrt() * torch.randn_like(x0_init)
    # et = model(xt_with_grad, t)
    # if et.size(1) == 6:
    #     var = torch.exp(et[:, 3:])
    #     et = et[:, :3]
    # x0_init = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
    # x0_init = torch.randn_like(x)
    # x0_init = x0_init.clamp(-1, 1)
    x0 = None
    for m in range(M):
        with torch.enable_grad():
            x0_t_with_grad = x0_init.clone().requires_grad_(True)
            optimizer = optim.AdamW([x0_t_with_grad], lr=vae_lr)
            # optimizer = optim.SGD([x0_t_with_grad], lr=vae_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_iters, eta_min=1e-5)
            for steps_n in tqdm(range(optimize_iters)):
                optimizer.zero_grad()
                loss_likelihood = torch.sum((y_0-H_funcs.forward(x0_t_with_grad))**2) # MAP
                # loss_likelihood = torch.linalg.norm(y_0-H_funcs.forward(x0_t_with_grad))
                i = int(noise_t)
                # noise_t -= 1
                # prior
                t = (torch.ones(n) * i).to(x.device)
                # at = alphas_cumprod[i]
                at = compute_alpha(b, t.long())
                # score
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
                # loss_prior = w_prior * torch.sum((x0_t - x0_t_with_grad)**2)
                loss_prior = w_prior * torch.sum(score.detach()/N * x0_t_with_grad)
                # 
                # w_prior *= 1.1
                loss = loss_likelihood + loss_prior
                # print(loss_likelihood / loss_prior)
                loss.backward()
                optimizer.step()
                scheduler.step()  # 在每次迭代后更新 lr
        x0_t = x0_t_with_grad.detach()
        xs = [x0_t.to('cpu')]
        x0_preds = [x0_t.to('cpu')]
        # print(xs)
        # print(x0_preds)
        # print(11111111111111111111)
        xt_next = x0_t
        # 接一个DPS
        betas = b.cpu().numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )

        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            # seq_next_next = [-1, -1] + list(seq[:-2])
            # t = (torch.ones(n) * int(renoise_t)).to(x.device)
            # if renoise_t > 0:
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
                    # print(var)
                    # 计算var                   
                    model_var_values = var
                    min_log = posterior_log_variance_clipped
                    max_log = np.log(betas)
                    # print(t[0])
                    min_log = extract_and_expand(min_log, t[0].long(), var)
                    max_log = extract_and_expand(max_log, t[0].long(), var)

                    # The model_var_values is [-1, 1] for [min_var, max_var]
                    frac = (model_var_values + 1.0) / 2.0
                    model_log_variance = frac * max_log + (1-frac) * min_log
                    model_variance = torch.exp(model_log_variance * 0.5)
                    # print(model_variance)
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
                # xt_next = beta_t * at_next.sqrt() / (1-at) * x0_t + (1 - at_next) / (1 - at) * alpha_t.sqrt() * xt + 1.0 * model_variance * torch.randn_like(x0_t) - xi * grad
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


def only_ps_generalized_steps(x, x_map, seq, model, b, H_funcs, y_0, sigma_0, xi=10.0, optimize_iters=60, vae_lr=0.5, w_prior=2.0, noise_t=10, renoise_t=0, M=1, ps_method='dps', cls_fn=None, classes=None):
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    #setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    t = (torch.ones(n) * seq[-1]).to(x.device)
    at = compute_alpha(b, t.long())
    noise = torch.randn_like(x)
    x_T = noise * (1 - at).sqrt()
    xt = x_T
    
    # 先做MAP
    x0_init = H_funcs.H_pinv(y_0).view(*x.shape)
    # i = int(50)
    # # prior
    # t = (torch.ones(n) * i).to(x.device)
    # # at = alphas_cumprod[i]
    # at = compute_alpha(b, t.long())
    # # score
    # xt_with_grad = at.sqrt() * x0_init + (1-at).sqrt() * torch.randn_like(x0_init)
    # et = model(xt_with_grad, t)
    # if et.size(1) == 6:
    #     var = torch.exp(et[:, 3:])
    #     et = et[:, :3]
    # x0_init = (xt_with_grad - et * (1 - at).sqrt()) / at.sqrt()
    # x0_init = torch.randn_like(x)
    # x0_init = x0_init.clamp(-1, 1)
    x0 = None
    x0_t = x_map
    xt_next = x0_t
    # 接一个DPS
    betas = b.cpu().numpy()
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # log calculation clipped because the posterior variance is 0 at the
    # beginning of the diffusion chain.
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    )

    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        # seq_next_next = [-1, -1] + list(seq[:-2])
        # t = (torch.ones(n) * int(renoise_t)).to(x.device)
        # if renoise_t > 0:
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
                # print(var)
                # 计算var                   
                model_var_values = var
                min_log = posterior_log_variance_clipped
                max_log = np.log(betas)
                # print(t[0])
                min_log = extract_and_expand(min_log, t[0].long(), var)
                max_log = extract_and_expand(max_log, t[0].long(), var)

                # The model_var_values is [-1, 1] for [min_var, max_var]
                frac = (model_var_values + 1.0) / 2.0
                model_log_variance = frac * max_log + (1-frac) * min_log
                model_variance = torch.exp(model_log_variance * 0.5)
                # print(model_variance)
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
            # xt_next = beta_t * at_next.sqrt() / (1-at) * x0_t + (1 - at_next) / (1 - at) * alpha_t.sqrt() * xt + 1.0 * model_variance * torch.randn_like(x0_t) - xi * grad
            xt = xt_next.detach()
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))
    x0 = xt_next.to('cpu')
    x0_preds = [x0]
    xs = [x0]
    return xs, x0_preds


# x: 初始噪声；seq: 时间下标序列；model: 扩散模型；betas: beta序列；H_funcs: 观测矩阵；y_0: 观测；sigma_0: 观测噪声标准差；clas_fn: 分类器；classes: 类别
def ddnm(x, seq, model, b, H_funcs, y_0, sigma_0, cls_fn=None, classes=None):
    # with torch.no_grad():
    #initialize x_T as given in the paper
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    
    #setup iteration variables
    # x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
    # y_upsampling = H_funcs.upsampling(y_0)
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    betas = b.cpu().numpy()
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # log calculation clipped because the posterior variance is 0 at the
    # beginning of the diffusion chain.
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    )
    t = (torch.ones(n) * seq[-1]).to(x.device)
    # alpha_t和alpha_{t-1}
    at = compute_alpha(b, t.long())
    noise = torch.randn_like(x)
    x_T = noise * (1 - at).sqrt()
    xt = x_T
    eta=0.15
    N = 1
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        if cls_fn == None:
            et = model(xt, t)
        else:
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
        
        if et.size(1) == 6:
            var = torch.exp(et[:, 3:])
            et = et[:, :3]
            model_var_values = var
            min_log = posterior_log_variance_clipped
            max_log = np.log(betas)
            # print(t[0])
            min_log = extract_and_expand(min_log, t[0].long(), var)
            max_log = extract_and_expand(max_log, t[0].long(), var)

            # The model_var_values is [-1, 1] for [min_var, max_var]
            frac = (model_var_values + 1.0) / 2.0
            model_log_variance = frac * max_log + (1-frac) * min_log
            model_variance = torch.exp(model_log_variance * 0.5)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        # calcultate mu and sigma in DDNM
        sigma_t = (1 - at_next**2).sqrt()
        sigma_y = sigma_0
        if sigma_t[0,0,0,0] >= at_next[0,0,0,0]*sigma_y:
            lambda_t = 1.
            gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
        else:
            lambda_t = (sigma_t)/(at_next*sigma_y)
            gamma_t = 0.
        # gamma_t = (1-at_next).sqrt()
        gamma_t = 1

        x0_t_hat = x0_t + lambda_t * H_funcs.H_pinv(y_0 - H_funcs.forward(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

        # different from the paper, we use DDIM here instead of DDPM
        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)
        xt = xt_next

        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return xs, x0_preds