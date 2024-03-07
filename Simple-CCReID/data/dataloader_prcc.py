import os
import random
import numpy as np
import torch
import functools
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDatasetPrcc(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, config, transform=None):
        self.generated_data_path = config.DATA.GEN_PATH
        self.thresholds = config.CL.THRESHOLDS

        self.percentages = config.CL.PERCENTAGES
        self.dataset = dataset
        self.transform = transform
        self.use_generated_data = config.DATA.USE_GENERATED
        self.data_type = config.DATA.DATASET
        self.same_img = config.DATA.SAME_IMG
        self.epoch_count = 0
        self.pid2clothes = {}
        self.img2genimg = {}
        if self.use_generated_data:
            for cloth_id in os.listdir(self.generated_data_path):
                for person_id in os.listdir(os.path.join(self.generated_data_path, cloth_id)):
                    if person_id not in self.pid2clothes:
                        self.pid2clothes[person_id] = [cloth_id]
                    else:
                        self.pid2clothes[person_id].append(cloth_id)
                    for img_name in os.listdir(os.path.join(self.generated_data_path, cloth_id, person_id)):
                        if (person_id + img_name) not in self.img2genimg:
                            self.img2genimg[person_id + img_name] = [cloth_id]
                        else:
                            self.img2genimg[person_id + img_name].append(cloth_id)
            for pid in self.pid2clothes:
                self.pid2clothes[pid] = random.sample(self.pid2clothes[pid], len(self.pid2clothes[pid]))

            pids = sorted([pid for pid in self.pid2clothes.keys()], key=lambda x: int(x))
            self.label2pid = {label: pid for label, pid in enumerate(pids)}
        self.dataset = dataset


    def update_epoch_count(self):
        self.epoch_count += 1
        print(f"Updated epoch:{self.epoch_count}")
        print(f"Current percentage:{self._decide_percentage()}")

    def _decide_percentage(self):
        for i, threshold in enumerate(self.thresholds):
            if self.epoch_count <= threshold:
                return self.percentages[i]
        return 1.

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_info = self.dataset[index]
        if len(data_info) == 4:
            img_path, label_pid, camid, clothes_id = self.dataset[index]
        else:
            img_path, label_pid, camid, clothes_id, clothes2label = self.dataset[index]
        img_name = img_path.split("/")[-1]
        img = read_image(img_path)
        if self.use_generated_data:
            pid_img_name = self.label2pid[label_pid] + img_name
            list_clothes = self.pid2clothes[self.label2pid[label_pid]]
            if self.same_img:
                if pid_img_name in self.img2genimg:
                    list_clothes = self.img2genimg[pid_img_name]

            final_index = max(1, int(len(list_clothes) * self._decide_percentage()))
            new_clothes = random.choice(list_clothes[:final_index])
            gen_imgs_path = os.path.join(self.generated_data_path, new_clothes, self.label2pid[label_pid])
            if not self.same_img or pid_img_name not in self.img2genimg:
                img_name = random.choice(os.listdir(gen_imgs_path))
            path = os.path.join(gen_imgs_path, img_name)
            gen_img = read_image(path)

        if self.transform is not None:
            img = self.transform(img)
            if self.use_generated_data:
                gen_img = self.transform(gen_img)
        if self.use_generated_data:
            return img, gen_img, label_pid, camid, clothes_id
        return img, label_pid, camid, clothes_id


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 dataset,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id
        else:
            return clip, pid, camid