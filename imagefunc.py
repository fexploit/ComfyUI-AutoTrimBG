import torch
import numpy as np
from PIL import Image, ImageDraw
from .imagefunc import *

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(p_image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(p_image).astype(np.float32) / 255.0).unsqueeze(0)

def image2mask(p_image: Image) -> torch.Tensor:
    if p_image.mode == "RGB":
        p_image = p_image.convert("L")
    return torch.from_numpy(np.array(p_image).astype(np.float32) / 255.0).unsqueeze(0)

def mask2image(t_mask: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8), mode="L")

def draw_rect(image, x, y, width, height, line_color, line_width):
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, x + width, y + height), outline=line_color, width=line_width)
    return image

def log(message, message_type='info'):
    if message_type == 'info':
        print(f"INFO: {message}")
    elif message_type == 'warning':
        print(f"WARNING: {message}")
    elif message_type == 'error':
        print(f"ERROR: {message}")
    elif message_type == 'finish':
        print(f"FINISH: {message}")