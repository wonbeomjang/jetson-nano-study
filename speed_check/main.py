import time

import torch
from torchvision.models.resnet import resnet18, ResNet18_Weights
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    image = torch.rand((1, 3, 256, 256)).to(device)

    start_time = time.time()
    pbar = tqdm(range(1000), total=1000)

    for i in pbar:
        model(image)
    end_time = time.time()

    print(end_time - start_time)