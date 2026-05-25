import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from third_party.ldm.models.diffusion.ddim import DDIMSampler
from third_party.ldm.util import instantiate_from_config
from torch.utils.data import Dataset, DataLoader
import tqdm

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    print(ckpt)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if 'global_step' in pl_sd.keys():
            global_step = pl_sd["global_step"]
        else:
            global_step = None
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


def get_model(resume='models/ldm/ffhq256/model.ckpt'):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)
    ckpt = resume

    if not os.path.exists(resume):
        raise ValueError("Cannot find {}".format(resume))
    logdir = '/'.join(resume.split('/')[:-1])
    # print(logdir)
    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))

    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    # cli = OmegaConf.from_dotlist(unknown)
    # print(configs)
    config = OmegaConf.merge(*configs)

    gpu = True
    eval_mode = True
    # print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    return model, config

if __name__ == '__main__':
    # model, config = get_model(resume='models/ldm/ffhq256/model.ckpt') # ffhq256
    # model, config = get_model(resume='models/ldm/celeba256/model.ckpt') # celeba256
    # model, config = get_model(resume='models/ldm/lsun_beds256/model.ckpt') # lsun_beds256
    model, config = get_model(resume='models/ldm/cin256/model.ckpt') # ffhq256
    print(model)
    # print(config)
