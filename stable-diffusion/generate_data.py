from time import sleep
import copy
import os

import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import InpaintingDataLoaderPrcc
from diffusers import StableDiffusionInpaintPipeline
from data.utils import threaded


# @threaded
def save_images(root, buffer_images):
    for images, image_paths, cloth_ids, ws, hs, masks, original_images in buffer_images:
        for i, img in enumerate(images):
            dir_path = os.path.join(root, "prcc", cloth_ids[i], image_paths[i].split("/")[0])
            os.makedirs(dir_path, exist_ok=True)
            final_path = os.path.join(dir_path, image_paths[i].split("/")[1].split(".")[0] + ".png")
            img = img.resize((ws[i], hs[i]))
            mask = masks[i]
            original_image = original_images[i]
            img = np.array(img)
            mask = Image.fromarray(mask.squeeze().numpy().astype(np.uint8)).resize((ws[i], hs[i]))
            original_image = Image.fromarray((original_image*255).squeeze().numpy().astype(np.uint8)).resize((ws[i], hs[i]))
            # original_image.show()
            original_image = np.array(original_image)
            mask = np.expand_dims(np.array(mask), axis=-1)
            img = mask*img + (1-mask)*original_image
            Image.fromarray(img.astype(np.uint8)).save(final_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--original_images_path", required=True, type=str)
    parser.add_argument("--clothes_description_path", required=True, type=str)
    parser.add_argument("--masks_path", required=True, type=str)
    parser.add_argument("--output_directory_path", required=True, type=str)
    parser.add_argument("--use_discriminator", default=False, required=False, type=bool)
    args = parser.parse_args()


    image_path = args.original_images_path
    mask_path = args.masks_path
    root = args.output_directory_path
    if args.use_discriminator:
        root = os.path.join(root, "disc")

    data_generator = InpaintingDataLoaderPrcc(args)
    batch_size = 5
    ddim_steps = 50
    scale = 7
    dataloader = DataLoader(data_generator, batch_size=batch_size, num_workers=6, shuffle=False, drop_last=False)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        revision="fp16",
        device_map='auto',
        safety_checker=None
    )

    height = 768
    width = 256
    jobs = []
    buffer_images = []
    length_dataset = len(dataloader.dataset)
    with torch.no_grad():
        for i, (original_image, prompt, mask, image_path, cloth_id, label_cloth_id, original_w, original_h) in enumerate(tqdm(dataloader)):
            if args.use_discriminator:
                gen_image = pipe(prompt=list(prompt), image=original_image, mask_image=mask, height=height, width=width, num_inference_steps=ddim_steps,
                 discriminator=pipe.discriminator, clothes_ids=label_cloth_id).images
            else:
                gen_image = pipe(prompt=list(prompt), image=original_image, mask_image=mask, height=height, width=width,
                                 num_inference_steps=ddim_steps).images
            buffer_images.append((gen_image, image_path, cloth_id, original_w, original_h, mask.swapaxes(1, -1).swapaxes(1, 2), original_image.swapaxes(1, -1).swapaxes(1, 2)))

            if len(buffer_images) >= 1:
                save_images(root, copy.deepcopy(buffer_images))
                buffer_images = []
                # break
    if len(buffer_images) >= 1:
        save_images(root, copy.deepcopy(buffer_images))
        buffer_images = []
        # break
    sleep(10)

