import torch

class LightningModule(torch.nn.Module):
    def save_hyperparameters(self, *args, **kwargs):
        pass

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model = cls(*args, **kwargs)
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)
        return model

class LightningDataModule:
    pass

class Trainer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Lightning Trainer is unavailable; this stub is for inference only.")
