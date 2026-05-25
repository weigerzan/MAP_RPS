import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--sigma_0", type=float, required=True, help="Sigma_0"
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, help="Step-size"
    )
    parser.add_argument(
        "--lam", type=float, default=1.0, help="Step-size"
    )
    parser.add_argument(
        "--optimize_iters", type=int, default=60, help="Number of optimization iterations"
    )
    parser.add_argument(
        "--vae_lr", type=float, default=0.5, help="Step-size"
    )
    parser.add_argument(
        "--w_prior", type=float, default=2.0, help="Weight for prior term"
    )
    parser.add_argument(
        "--eta_min", type=float, default=1e-5, help="Weight for prior term"
    )
    parser.add_argument(
        "--noise_t", type=int, default=10, help="Noise timestep"
    )
    parser.add_argument(
        "--renoise_t", type=int, default=0, help="Renoise timestep"
    )
    parser.add_argument(
        "--M", type=int, default=1, help="M avg"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="map_rps",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ps_method",
        type=str,
        default="dps",
    )
    parser.add_argument(
        "--stable", action="store_true", help="Using stablized update in lmap_rps"
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )

    args = parser.parse_args()
    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        runner.sample()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
