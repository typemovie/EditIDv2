import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from torchvision import transforms


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, size=512, t_drop_rate=0.05, i_drop_rate=0.05,
                 ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]
        # [
        #   {"id_image": "1737712399730-1771bad8-83e0-47c1-8d66-09c086c38aa4.jpeg", "text": "A snake"},
        #   {"id_image": "face_recognition_1.png", "image":"","text": "A female"}
        # ]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        id_image = item["id_image"]
        image = item["image"]

        # read image
        image = Image.open(os.path.join(self.image_root_path, image))
        image = self.transform(image.convert("RGB"))
        id_image = Image.open(os.path.join(self.image_root_path, id_image))
        id_image = np.array(id_image)

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        return {
            "image": image,
            "text": text,
            "id_image": id_image,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    texts = [example["text"] for example in data]
    id_images = torch.cat([torch.tensor(example["id_image"]) for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "texts": texts,
        "id_images": id_images,
        "drop_image_embeds": drop_image_embeds
    }


def pulid_loader(num_workers, train_batch_size, **args):
    dataset = MyDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, collate_fn=collate_fn)
