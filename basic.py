import numpy as np
import torch
from tqdm import tqdm

from data import ImageNetDataModule
from models import ResNet18
from attack import SIA


if __name__ == '__main__':
    CNT = 10
    BS = 16

    dm = ImageNetDataModule(root_dir='./data/imagenet/data', class_index_file='./data/imagenet/class_index.json')
    data_loader = dm.get_data_loader(batch_size=BS, shuffle=True)
    model = ResNet18()
    model.eval()
    model.hook_middle_representation()
    atk = SIA(model, eps=8/255, alpha=2/225, steps=200, gamma=64, random_start=True)

    suc = 0
    cnt = CNT
    tags = []

    for images, labels in tqdm(data_loader):
        # adv_images = atk(images, labels)
        # outputs = model(adv_images)
        outputs = model(images)

        _, pre = torch.max(outputs.data, 1)

        tags += ['adv_' + dm.idx2label[i] for i in pre]
        a = (pre == labels.to(model.device)).cpu().numpy()
        suc += np.size(a) - np.count_nonzero(a)

        cnt -= 1
        if cnt == 0:
            break

    print('- ASR:', suc / (BS * CNT))
