import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data

FIXED_PAIRS = [
            (0, 1), (10, 20), (50, 60), (100, 110), (5, 15), 
            (30, 40), (70, 80), (90, 100), (12, 22), (45, 55),
            (150, 160), (200, 210), (250, 260), (300, 310), (350, 360),
            (3,6), (8, 21), (38, 7), (23, 79), (47, 51),
            (2, 347), (6, 17), (322, 327), (333, 344), (4, 301)
        ]

FIXED_PAIRS_SHORT = [
            (0, 1), (10, 20), (50, 60), (100, 110), (5, 15), 
            (30, 40), (70, 80), (90, 100), (12, 22), (45, 55),
            (150, 160), (200, 210), (250, 260), (300, 310), (350, 360)
        ]

def subsample(data, targets, num_data, num_classes):
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
    new_targets = targets[idx][:num_data]

    return torch.utils.data.TensorDataset(new_data, new_targets)


def new_encoder(M):
    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.Softmax(dim=1),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.Softmax(dim=1),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2 * M),
    )
    return encoder_net

def new_decoder(M):
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.BatchNorm2d(32),
        nn.Softmax(dim=1),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(32),
        nn.Softmax(dim=1),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(16),
        nn.Softmax(dim=1),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )
    return decoder_net
