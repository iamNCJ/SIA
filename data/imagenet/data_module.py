import json
from typing import Tuple, List, Dict

import timm
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ImageSubFolder(dsets.ImageFolder):
    """
    Class of ImageFolder that only loads images from the specified classes.
    """
    def __init__(self, root, transform=None, target_transform=None, selected_classes=None):
        self.selected_classes = selected_classes
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        if self.selected_classes is None:
            return super().find_classes(directory)
        classes = self.selected_classes.copy()
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def image_folder_custom_label(root, transform, idx2label, selected_classes=None):
    old_data = ImageSubFolder(root=root, transform=transform, selected_classes=selected_classes)
    old_classes = old_data.classes

    label2idx = {}
    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = ImageSubFolder(root=root, transform=transform,
                              target_transform=lambda x: idx2label.index(old_classes[x]),
                              selected_classes=selected_classes)
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


class ImageNetDataModule:
    def __init__(self, root_dir: str, class_index_file: str, selected_classes: list = None):
        self.label2idx = json.load(open(class_index_file))
        self.idx2label = [self.label2idx[str(k)][1] for k in range(len(self.label2idx))]
        model = timm.create_model('resnet18', pretrained=True)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]

            # Using normalization for Inception v3.
            # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                          std=[0.229, 0.224, 0.225])

            # However, DO NOT USE normalization transforms here.
            # Torchattacks only supports images with a range between 0 and 1.
            # Thus, please refer to the model construction section.
        ])
        self.imagenet_data = image_folder_custom_label(
            root=root_dir,
            transform=transform,
            idx2label=self.idx2label,
            selected_classes=selected_classes
        )

    def get_idx2label(self) -> list:
        return self.idx2label

    def get_label2idx(self) -> dict:
        return self.label2idx

    def get_data_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return torch.utils.data.DataLoader(self.imagenet_data, batch_size=batch_size, shuffle=shuffle, num_workers=16)


if __name__ == '__main__':
    dm = ImageNetDataModule(root_dir='./data', class_index_file='./class_index.json', selected_classes=['alp'])
    data_loader = dm.get_data_loader(batch_size=16, shuffle=False)
    images, labels = iter(data_loader).next()
    print(images.shape)
    print(labels)
    print(dm.idx2label[labels[0]])
