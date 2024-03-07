import json
import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class InpaintingDataLoaderPrcc(Dataset):
    def __init__(self, args):
        self.h = 768
        self.w = 256
        cloth_descriptions_path = args.clothes_description_path


        self.subject_path = args.original_images_path
        self.mask_path = args.masks_path
        self.image_list = []

        output_dir = args.output_directory_path
        os.makedirs(output_dir, exist_ok=True)

        self.clothes_descriptions_path = cloth_descriptions_path
        with open(self.clothes_descriptions_path) as file:
            self.clothes = json.load(file)
        cloth_ids = set(self.clothes.keys())
        cloth_ids = tuple(cloth_ids)
        self.dict_clothes_ids = {cloth_id: i for i, cloth_id in enumerate(cloth_ids)}
        list_subjects = os.listdir(self.subject_path)
        list_subjects = sorted(list_subjects)
        for subject in list_subjects:
            for image in os.listdir(os.path.join(self.subject_path, subject)):
                if image[0] == 'C':
                    current_cloth_id = f'C_{subject}'
                else:
                    current_cloth_id = f'AB_{subject}'
                sample_cloths = random.sample(cloth_ids, min(30, len(cloth_ids)))
                count = 0
                for cloth_id in sample_cloths:
                    if cloth_id != current_cloth_id:
                        self.image_list.append(
                                (f'{subject}/{image}', f'{subject}/{image.split(".")[0]}.png', cloth_id))
                        count += 1
                    if count == 5:
                            break


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index][0]
        image = Image.open(os.path.join(self.subject_path, image_path))
        w, h = image.size
        image = image.resize((self.w, self.h))
        mask_img = self.image_list[index][1]
        mask = np.array(Image.open(f"{self.mask_path}/{mask_img}").resize((self.w, self.h)))
        cloth_id = self.image_list[index][-1]

        items = [word.lower().replace(",", "") for word in self.clothes[cloth_id]]  # [:3]
        items = [word for word in items if len(word) > 2]
        prompt = " ".join(items)
        returned_cloth_id = cloth_id

        cloth_id = self.image_list[index][-1]
        mask = np.expand_dims(mask, axis=-1)
        mask = (mask>0) *255.
        image = np.array(image, dtype=np.float32)
        image = image.swapaxes(0, -1).swapaxes(1, -1)
        mask = mask.swapaxes(0, -1).swapaxes(1, -1)
        image /= 255.
        mask /= 255.
        return image, prompt, mask, image_path, returned_cloth_id, self.dict_clothes_ids[cloth_id], w, h



