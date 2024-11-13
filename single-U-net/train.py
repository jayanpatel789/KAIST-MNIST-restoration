"""
Script for the training of the U-Net model 
"""

import unet_model
import torchvision
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
import os, sys
from data2 import FingerprintDataset


def train(model, device, train_loader, optimizer, epoch, version):
    model.train()
    for batch_idx, images in enumerate(train_loader):
        input = images["input"]
        target = images["target"]
        input, target = input.to(device), target.to(device)
        output = model(input)
        # print('target shape : ', target.shape)
        # print('output shape : ', output.shape)
        train_loss = nn.MSELoss()(output, target)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
              print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(input), len(train_loader.dataset),
                    100.*batch_idx/len(train_loader), train_loss.item()))
    
    # dir = f"/home/kiss2023/workspace/Jayan/models/version{version}"
    # if os.path.exists(dir):
    #     raise Exception("This model version has already been trained. Increase the version number.")
    makedir(f"/home/kiss2023/workspace/Jayan/models/version{version}")
    modelname = str(f"/home/kiss2023/workspace/Jayan/models/version{version}/fingerprintmodel{version}-{epoch}.pt")
    torch.save(model, modelname)

# Logs the model and model parts you used into a text file.
def log(version):
    files = ["unet_model.py", "unet_parts.py", "train.py"]
    currenttime = datetime.now().strftime('%d-%m_%H:%M')
    for file in files:
        with open(file, 'r') as openfile:
            filecode = openfile.read()
            filename = "v" + version + "_" + currenttime + ".txt"
            if "parts" in file:
                log_path = "/home/kiss2023/workspace/Jayan/logs/parts/"
            elif "model" in file:
                log_path = "/home/kiss2023/workspace/Jayan/logs/models/"
            else:
                log_path = "/home/kiss2023/workspace/Jayan/logs/trainers/"
            makedir(log_path)
            with open(log_path + filename, 'w') as outputfile:
                outputfile.write(filecode)

def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)
              
def main():
    args = sys.argv
    if len(args) != 2:
        sys.exit("Usage: (CUDA_VISIBLE_DEVICES=[gpu]), python train.py [version])")
    version = args[1]

    log(version)

    trainset = FingerprintDataset(
        mode="train",
        levels="easy",
        ratio=0.9,
        imagetype="bmp",
        transform=transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor()
        ])
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = unet_model.UNet(n_channels=1, n_classes=1)
    #model = torch.load("/home/kiss2023/workspace/Jayan/U-Net3/models/fingerprintmodel6-7.pt")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.005)

    for epoch in range(10):
        train(model, device, train_loader, optimizer, epoch, version)
        scheduler.step()

if __name__ == "__main__":
    main()