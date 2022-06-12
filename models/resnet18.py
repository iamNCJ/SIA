import torch
from torch import nn
from torchvision import models

from data import ImageNetDataModule


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


class ResNet18(nn.Module):
    """
    ResNet18 Module that can extract features from the middle of the network
    """
    def __init__(self, use_cuda=True):
        super(ResNet18, self).__init__()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model = nn.Sequential(
            norm_layer,
            models.resnet18(pretrained=True)
        ).to(self.device)
        self.features = {}

    def hook_middle_representation(self):
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        self.model[1].avgpool.register_forward_hook(get_features('feats_plr'))
        # self.model[1].layer4.register_forward_hook(get_features('feats_4'))
        # self.model[1].layer3.register_forward_hook(get_features('feats_3'))
        # self.model[1].layer2.register_forward_hook(get_features('feats_2'))
        # self.model[1].layer1.register_forward_hook(get_features('feats_1'))
        # self.model[1].conv1.register_forward_hook(get_features('feats_0'))

    def forward(self, x):
        return self.model(x.to(self.device))


if __name__ == '__main__':
    dm = ImageNetDataModule(root_dir='../data/imagenet/data', class_index_file='../data/imagenet/class_index.json')
    data_loader = dm.get_data_loader(batch_size=16, shuffle=False)
    images, labels = iter(data_loader).next()
    print(labels)
    print(images.shape)
    model = ResNet18()
    model.hook_middle_representation()
    model.forward(images)
    print(model.features.keys())
    print(model.features['feats_plr'].shape)
