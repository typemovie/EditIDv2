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

        # 确保json_files和image_root_paths都是列表，且长度一致
        if isinstance(json_file, str):
            json_files = [json_file]
        else:
            json_files = json_file
        if isinstance(image_root_path, str):
            image_root_paths = [image_root_path]
        else:
            image_root_paths = image_root_path

        assert len(json_files) == len(image_root_paths), \
            f"json_files length ({len(json_files)}) must match image_root_paths length ({len(image_root_paths)})"

        self.json_files = json_files
        self.image_root_paths = image_root_paths

        # 加载所有数据源的数据
        self.data = []
        self.data_source_mapping = []  # 记录每个数据项对应的数据源索引

        for source_idx, (json_file, image_root_path) in enumerate(zip(json_files, image_root_paths)):
            source_data = json.load(open(json_file))
            self.data.extend(source_data)
            # 记录每个数据项来自哪个数据源
            self.data_source_mapping.extend([source_idx] * len(source_data))

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, idx):
        item = self.data[idx]
        source_idx = self.data_source_mapping[idx]  # 获取对应的数据源索引
        image_root_path = self.image_root_paths[source_idx]  # 使用对应的图片根路径

        text = item["text"]
        id_image = item["id_image"]
        image = item["image"]

        # read image - 使用对应数据源的图片根路径
        image = Image.open(os.path.join(image_root_path, image))
        image = self.transform(image.convert("RGB"))
        id_image = Image.open(os.path.join(image_root_path, id_image))
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
            "drop_image_embed": drop_image_embed,
            "source_idx": source_idx  # 可选：返回数据源索引用于调试
        }

    def __len__(self):
        return len(self.data)

    def get_source_info(self):
        """返回数据源的统计信息"""
        source_counts = {}
        for i, source_idx in enumerate(self.data_source_mapping):
            if source_idx not in source_counts:
                source_counts[source_idx] = 0
            source_counts[source_idx] += 1

        print("Data source statistics:")
        for source_idx, count in source_counts.items():
            print(f"  Source {source_idx} ({self.json_files[source_idx]}): {count} samples")
        print(f"Total samples: {len(self.data)}")


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
    # 打印数据源统计信息
    dataset.get_source_info()
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, collate_fn=collate_fn)
