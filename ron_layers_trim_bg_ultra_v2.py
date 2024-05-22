import torch
import numpy as np
from PIL import Image
from .imagefunc import *

NODE_NAME = 'RonLayersTrimBgUltraV2'

class RonLayersTrimBgUltraV2:
    def __init__(self):
        self.input_image = None
        self.input_mask = None
        self.output_image = None
        self.output_mask = None
        self.crop_box = None
        self.box_preview = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE")
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = "trim_and_crop_by_mask"
    CATEGORY = "RonLayers/TrimBg"

    def trim_and_crop_by_mask(self, image, mask, padding):
        ret_images = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        if mask.shape[0] > 1:
            log(f"Warning: Multiple mask inputs, using the first.", message_type='warning')
            mask = torch.unsqueeze(mask[0], 0)

        image_pil = tensor2pil(image).convert('RGBA')
        mask_pil = tensor2pil(mask).convert('L')

        masked_image = Image.composite(image_pil, Image.new("RGBA", image_pil.size), mask_pil)
        bbox = masked_image.getbbox()

        if bbox:
            x1, y1, x2, y2 = bbox
            x1 = x1 - padding if x1 - padding > 0 else 0
            y1 = y1 - padding if y1 - padding > 0 else 0
            x2 = x2 + padding if x2 + padding < image_pil.width else image_pil.width
            y2 = y2 + padding if y2 + padding < image_pil.height else image_pil.height
            crop_box = (x1, y1, x2, y2)
            cropped_image = Image.composite(image_pil.crop(crop_box), Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0)), mask_pil.crop(crop_box))
            cropped_mask = mask_pil.crop(crop_box)
            preview_image = draw_rect(tensor2pil(mask).convert('RGB'), x1, y1, x2 - x1, y2 - y1, line_color="#00F000",
                                      line_width=(x2 - x1 + y2 - y1) // 200)
            ret_images.append(pil2tensor(cropped_image.convert("RGBA")))
            ret_masks.append(image2mask(cropped_mask))
        else:
            ret_images.append(image)
            ret_masks.append(mask)
            crop_box = None
            preview_image = tensor2pil(mask).convert('RGB')

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), crop_box, pil2tensor(preview_image))

    def run(self, image, mask, padding):
        self.input_image = image
        self.input_mask = mask
        self.output_image, self.output_mask, self.crop_box, self.box_preview = self.trim_and_crop_by_mask(image=self.input_image, mask=self.input_mask, padding=padding)
        return self.output_image, self.output_mask, self.crop_box, self.box_preview

NODE_CLASS_MAPPINGS = {
    "RonLayers/TrimBg: RonLayersTrimBgUltraV2": RonLayersTrimBgUltraV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RonLayers/TrimBg: RonLayersTrimBgUltraV2": "RonLayers/TrimBg: RonLayersTrimBgUltraV2"
}