import argparse
from utilities import Engine, save_image, TRT_LOGGER
from models import CLIP, UNet, VAE
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
import numpy as np
from PIL import Image
import torch
from math import floor
from polygraphy import cuda


_BATCHSIZE = 1
_MAX_SIZE = 512
_ENGINE_DIR = "anything/engine"
_HF_VAE_MODEL_NAME = "/yanbc/workspace/codes/img2img/src_models/anything-fp32/vae/"
_HF_TOKENIZER_NAME = "/yanbc/workspace/codes/img2img/src_models/anything-fp32/tokenizer/"
_HF_CLIPTEXT_NAME = "/yanbc/workspace/codes/img2img/src_models/anything-fp32/text_encoder/"
_N_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad anatomy, bad hands, text,error, missing fngers,extra digt ,fewer digits,cropped,worst quality ,low quality,normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet,NSFW, lowres,bad anatomy,text, error,nsfw,nsfw"
_SEED = 1631445808
_STRENGTH = 0.4
_GUIDANCE_SCALE = 12

torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)


def load_img(path: str) -> Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    max_edge = max(w, h)
    scale = max_edge / _MAX_SIZE
    if scale > 1:
        w, h = floor(w / scale), floor(h / scale)
    w, h = map(lambda x: x - x % 64, (w, h))        # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.LANCZOS)
    return image


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="path to image file")
    return p.parse_args()


class Img2Img:
    def __init__(self) -> None:
        strength = _STRENGTH
        denoising_steps = 50
        guidance_scale = _GUIDANCE_SCALE
        device = "cuda"
        max_batch_size = _BATCHSIZE
        engine_dir = _ENGINE_DIR
        batch_size = 1

        models = {
            # 'clip': CLIP(hf_token="", device=device, max_batch_size=max_batch_size),
            'unet_fp16': UNet(hf_token="", fp16=True, device=device, max_batch_size=max_batch_size),
            'vae': VAE(hf_token="", device=device, max_batch_size=max_batch_size)
        }
        engines = {}
        for model_name, obj in models.items():
            engine = Engine(model_name, engine_dir)
            engines[model_name] = engine
            engines[model_name].activate()

        autoencoder = AutoencoderKL.from_pretrained(_HF_VAE_MODEL_NAME).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(_HF_TOKENIZER_NAME)
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            skip_prk_steps=True,
            steps_offset=1,
        )
        text_encoder = CLIPTextModel.from_pretrained(_HF_CLIPTEXT_NAME).to(device)

        self.guidance_scale = guidance_scale
        self.device = device
        self.models = models
        self.engines = engines
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.autoencoder = autoencoder
        self.denoising_steps = denoising_steps
        self.strength = strength
        self.stream = cuda.Stream()
        self.batch_size = batch_size
        self.denoising_fp16 = True
        self.text_encoder = text_encoder

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

    def __encode_prompt(self, prompt: str) -> torch.Tensor:
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)

        text_input_ids_inp = cuda.DeviceView(ptr=text_input_ids.data_ptr(), shape=text_input_ids.shape, dtype=np.int32)

        attention_mask = torch.ones_like(text_input_ids, dtype=torch.int32, device=self.device)
        attention_mask_inp = cuda.DeviceView(ptr=attention_mask.data_ptr(), shape=attention_mask.shape, dtype=np.int32)
        text_embeddings = self.runEngine('clip', {"input_ids": text_input_ids_inp, "attention_mask": attention_mask_inp})['text_embeddings']

        n_prompt = _N_PROMPT
        max_length = text_input_ids.shape[-1]
        uncond_input_ids = self.tokenizer(
            n_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        uncond_input_ids_inp = cuda.DeviceView(ptr=uncond_input_ids.data_ptr(), shape=uncond_input_ids.shape, dtype=np.int32)
        uncond_embeddings = self.runEngine('clip', {"input_ids": uncond_input_ids_inp, "attention_mask": attention_mask_inp})['text_embeddings']

        pt_text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)
        return pt_text_embeddings

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        text_embeddings = self.text_encoder(text_input_ids)[0]

        n_prompt = _N_PROMPT
        max_length = text_input_ids.shape[-1]
        uncond_input_ids = self.tokenizer(
            n_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        uncond_embeddings = self.text_encoder(uncond_input_ids)[0]

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
        self.autoencoder.encoder = self.autoencoder.encoder.to(torch.float32)
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
        # Engine allocate buffers
        image_w, image_h = image.size
        for model_name, obj in self.models.items():
            self.engines[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(_BATCHSIZE, image_h, image_w), device=self.device)

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
        ).to(torch.float32)

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

    # image_path = "/yanbc/workspace/images/2017.jpeg"
    # # prompt = "mikasa in the dark, blood moon, twilight, jumping between buildings, immensely detailed"
    # # image_path = "output/sd-fp16-c-1-1262.png"
    # # prompt = "cute anime girl, highly detailed"
    # # image_path = "./input_image.jpg"
    # # prompt = "masterpiece, best quality, Anime style, a woman in a skirt and jacket posing for a picture with her hair blowing in the wind and smiling,"
    # prompt = "masterpiece, best quality, Anime style, a woman in a skirt and jacket posing for a picture with her hair blowing in the wind and smiling"

    # image_path = "/yanbc/workspace/images/25.jpg"
    # image_path = "./resized.jpg"
    # prompt = "masterpiece, best quality, Anime style, a woman holding a gun in front of a muggy muggy muggy muggy muggy muggy muggy muggy muggy muggy muggy muggy muggy muggy muggy muggy mug"
    image_path = "./resized_mimi.jpg"
    prompt = "masterpiece, best quality, Anime style, a woman in a black dress with a flowered top on her head and a ring on her finger"

    image = load_img(image_path)

    img2img = Img2Img()
    img2img.infer(prompt, image)
