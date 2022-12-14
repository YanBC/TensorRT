from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from math import floor


_MAX_SIZE = 1024
_SEED = 1631445808
_STRENGTH = 0.4
_GUIDANCE_SCALE = 12




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


model_id = "Linaqruf/anything-v3.0"
model_id = "/yanbc/workspace/codes/img2img/src_models/anything-fp32"
branch_name= "diffusers"
# branch_name = ""


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, revision=branch_name, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe = pipe.to("cuda")

# image_path = "./resized.jpg"
# image_path = "/yanbc/workspace/images/25.jpg"
image_path = "./resized_mimi.jpg"
prompt = "masterpiece, best quality, Anime style, a woman in a black dress with a flowered top on her head and a ring on her finger"
n_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad anatomy, bad hands, text,error, missing fngers,extra digt ,fewer digits,cropped,worst quality ,low quality,normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet,NSFW, lowres,bad anatomy,text, error,nsfw,nsfw"


init_image = load_img(image_path)

torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
pipe_out = pipe(prompt, init_image, strength=_STRENGTH, guidance_scale=_GUIDANCE_SCALE, negative_prompt=n_prompt)

image = pipe_out.images[0]

image.save("./pikachu_img2img_mimi.png")
