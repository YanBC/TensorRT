import argparse
from utilities import Engine, save_image, TRT_LOGGER
from models import CLIP, UNet, VAE
from transformers import CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
import numpy as np
from PIL import Image
import torch
from math import floor
from polygraphy import cuda


def load_img(path: str) -> Image:
    _MAX_SIZE = 512
    image = Image.open(path).convert("RGB")
    w, h = image.size
    max_edge = max(w, h)
    scale = max_edge / _MAX_SIZE
    if scale > 1:
        w, h = floor(w / scale), floor(h / scale)
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    new_image = Image.new(image.mode, (512, 512), (0,0,0))
    new_image.paste(image, (0,0))
    image = new_image

    return new_image


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="path to image file")
    return p.parse_args()


class Img2Img:
    def __init__(self) -> None:
        strength = 0.6
        denoising_steps = 50
        guidance_scale = 7.5
        device = "cuda"
        max_batch_size = 16
        engine_dir = "engine"
        batch_size = 1
        image_height = 512
        image_width = 512

        models = {
            'clip': CLIP(hf_token="", device=device, max_batch_size=max_batch_size),
            'unet_fp16': UNet(hf_token="", fp16=True, device=device, max_batch_size=max_batch_size),
            'vae': VAE(hf_token="", device=device, max_batch_size=max_batch_size)
        }
        engines = {}
        for model_name, obj in models.items():
            engine = Engine(model_name, engine_dir)
            engines[model_name] = engine
            engines[model_name].activate()
            engines[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=device)

        autoencoder = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",
            subfolder="vae").to(device)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            skip_prk_steps=True,
            steps_offset=1,
        )

        self.guidance_scale = guidance_scale
        self.device = device
        self.engines = engines
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.prompt = "mikasa in the dark, blood moon, twilight, jumping between buildings, immensely detailed"
        self.autoencoder = autoencoder
        self.denoising_steps = denoising_steps
        self.strength = strength
        self.stream = cuda.Stream()
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.denoising_fp16 = True

    def runEngine(self, model_name, feed_dict):
        engine = self.engines[model_name]
        return engine.infer(feed_dict, self.stream)

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        text_input_ids_inp = cuda.DeviceView(ptr=text_input_ids.data_ptr(), shape=text_input_ids.shape, dtype=np.int32)
        text_embeddings = self.runEngine('clip', {"input_ids": text_input_ids_inp})['text_embeddings']
        # bs_embed, seq_len, _ = text_embeddings.shape
        # text_embeddings = text_embeddings.repeat(1, 1, 1)
        # text_embeddings = text_embeddings.view(bs_embed * 1, seq_len, -1)

        max_length = text_input_ids.shape[-1]
        uncond_input_ids = self.tokenizer(
            "",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        uncond_input_ids_inp = cuda.DeviceView(ptr=uncond_input_ids.data_ptr(), shape=uncond_input_ids.shape, dtype=np.int32)
        uncond_embeddings = self.runEngine('clip', {"input_ids": uncond_input_ids_inp})['text_embeddings']

        pt_text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)
        return pt_text_embeddings

    def _image_preprocess(self, image: Image) -> torch.Tensor:
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    def _prepare_latents(self, init_image: torch.Tensor, timestep, device, generator=None):
        init_image = init_image.to(device=device, dtype=torch.float32)
        init_latents_dist = self.autoencoder.encode(init_image, return_dict=False)[0]
        init_latents = init_latents_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents
        # init_latents = torch.cat([init_latents] * 2, dim=0)

        noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=torch.float32)
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def _decode_latents(self, latents):
        latents = 1. / 0.18215 * latents
        sample_inp = cuda.DeviceView(ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32)
        images = self.runEngine('vae', {"latent": sample_inp})['images']
        return images

    def infer(self, prompt: str, image: Image):
        # Encode input prompt
        text_embeddings = self._encode_prompt(prompt)

        # Preprocess image
        init_image = self._image_preprocess(image)

        # Set timesteps
        self.scheduler.set_timesteps(self.denoising_steps, device=self.device)
        timesteps = self.get_timesteps(self.denoising_steps, self.strength)
        latent_timestep = timesteps[:1]

        # Prepare latent variables
        latents = self._prepare_latents(
                init_image=init_image,
                timestep=latent_timestep,
                device=self.device
        )

        # Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            timestep_model_input = t.to(dtype=torch.float32, device=self.device)

            sample_inp = cuda.DeviceView(
                    ptr=latent_model_input.data_ptr(),
                    shape=latent_model_input.shape,
                    dtype=np.float32
            )
            timestep_inp = cuda.DeviceView(
                    ptr=timestep_model_input.data_ptr(),
                    shape=timestep_model_input.shape,
                    dtype=np.float32,
            )
            embeddings_inp = cuda.DeviceView(
                    ptr=text_embeddings.data_ptr(),
                    shape=text_embeddings.shape,
                    dtype=np.float16
            )
            noise_pred = self.runEngine('unet_fp16', {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp})['latent']

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Post-processing
        images = self._decode_latents(latents)

        # Save image
        image_name_prefix = 'sd-'+('fp16' if self.denoising_fp16 else 'fp32')+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(1)]))+'-'
        save_image(images, "output", image_name_prefix)


if __name__ == "__main__":
    # args = get_args()
    # image_path = args.image

    image_path = "/yanbc/workspace/images/2017.jpeg"
    prompt = "mikasa in the dark, blood moon, twilight, jumping between buildings, immensely detailed"

    image = load_img(image_path)

    img2img = Img2Img()
    img2img.infer(prompt, image)
