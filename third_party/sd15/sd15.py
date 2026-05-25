from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch


class SD15:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.model_id = model_id
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.alphas_cumprod = self.pipe.scheduler.alphas_cumprod
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.unet.eval()
        self.vae.eval()
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
    
    def apply_model(self, z_t, t, prompt, cfg=1.5):
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            text_embeddings = self.text_encoder(**tokens).last_hidden_state
            with torch.autocast("cuda"):
                noise_pred_cond = self.unet(z_t, t, encoder_hidden_states=text_embeddings).sample
            if cfg == 1.0:
                return noise_pred_cond
            else:
                uncond_tokens = self.tokenizer(
                    [""] * z_t.shape[0], return_tensors="pt"
                ).to(self.device)
                uncond_embeddings = self.text_encoder(**uncond_tokens).last_hidden_state
                with torch.autocast("cuda"):
                    noise_pred_uncond = self.unet(z_t, t, encoder_hidden_states=uncond_embeddings).sample
                noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
                return noise_pred
    
    def encode_first_stage(self, x):
        with torch.autocast("cuda", dtype=torch.float16):
            latent = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
            return latent
    
    def decode_first_stage(self, z):
        with torch.autocast("cuda", dtype=torch.float16):
            recon_img = self.vae.decode(z / self.vae.config.scaling_factor).sample
        return recon_img