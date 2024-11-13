"""
For testing of Connected U Net
"""

import torchvision
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os, sys
from datetime import datetime
from data2 import FingerprintDataset

def test(model0, model1, device, test_loader, modelname, n_out_images=10):
    currenttime = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    model0.eval()
    model1.eval()
    dir = f"/home/kiss2023/workspace/Jayan/results/testout{modelname}/"
    makedir(dir)
    resultsfile = "/home/kiss2023/workspace/Jayan/results.txt"
    
    criterion = nn.MSELoss()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for i, images in enumerate(test_loader):
            input = images["input"]
            target = images["target"]
            input, target = input.to(device), target.to(device)
            inter = model0(input)
            feature_map = model0.get_feature_maps()
            model1.feature_map_in = feature_map
            output = model1(inter)

            if i < n_out_images:
                subjects = images["subjects"]
                damages = images["damages"]
                levels = images["levels"]
                imageresult = torch.stack([input[0], target[0], output[0]], dim=0)
                name = f"{subjects[0]}-{damages[0]}-{levels[0]}.png"
                filename = dir + name
                save_image(imageresult, filename)

            batch_loss_ps = criterion(output, target) # batch loss per sample
            total_loss += batch_loss_ps.item() * input.shape[0] # Multiply by batch_size to get total loss
            total_samples += input.shape[0] # Batch size
    
    mse_loss = total_loss / total_samples
    with open(resultsfile, 'a') as openfile:
         openfile.write(f"\n{currenttime}, Model: {modelname}, MSE Loss: {mse_loss}, Samples: {total_samples} Notes:")
    
def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def main():
    args = sys.argv
    if len(args) != 3:
        sys.exit("Usage: (CUDA_VISIBLE_DEVICES=[gpu], python test.py [version] [epoch])")
    version = args[1]
    epoch = args[2]
    modelname = str(version) + "-" + str(epoch)

    testset = FingerprintDataset(
        mode="test",
        levels="easy medium hard",
        ratio=0.9,
        imagetype="bmp",
        damagetypes=["CR"],
        transform=transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor()
        ])
    )
                                 
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0 = torch.load("/home/kiss2023/workspace/Jayan/models/version7/fingerprintmodel7-19.pt")
    model1 = torch.load(f"/home/kiss2023/workspace/Jayan/models/version{version}/fingerprintmodel{modelname}.pt")
    model0, model1 = model0.to(device), model1.to(device)

    test(model0, model1, device, test_loader, modelname)

# """
# For testing of single U Net
# """

# import torchvision
# import torch
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# import torch.nn.functional as F
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
# import os, sys
# from datetime import datetime
# from data2 import FingerprintDataset

# def test(model, device, test_loader, modelname, n_out_images=10):
#     currenttime = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
#     model.eval()
#     dir = f"/home/kiss2023/workspace/Jayan/results/testout{modelname}/"
#     makedir(dir)
#     resultsfile = "/home/kiss2023/workspace/Jayan/results.txt"
    
#     criterion = nn.MSELoss()
#     total_loss = 0
#     total_samples = 0

#     with torch.no_grad():
#         for i, images in enumerate(test_loader):
#             input = images["input"]
#             target = images["target"]
#             input, target = input.to(device), target.to(device)
#             output = model(input)

#             if i < n_out_images:
#                 subjects = images["subjects"]
#                 damages = images["damages"]
#                 levels = images["levels"]
#                 imageresult = torch.stack([input[0], target[0], output[0]], dim=0)
#                 name = f"{subjects[0]}-{damages[0]}-{levels[0]}.png"
#                 filename = dir + name
#                 save_image(imageresult, filename)

#             batch_loss_ps = criterion(output, target) # batch loss per sample
#             total_loss += batch_loss_ps.item() * input.shape[0] # Multiply by batch_size to get total loss
#             total_samples += input.shape[0] # Batch size
    
#     mse_loss = total_loss / total_samples
#     with open(resultsfile, 'a') as openfile:
#          openfile.write(f"\n{currenttime}, Model: {modelname}, MSE Loss: {mse_loss}, Samples: {total_samples} Notes:")
    
# def makedir(path):
# 	if not os.path.exists(path):
# 		os.makedirs(path)

# def main():
#     args = sys.argv
#     if len(args) != 3:
#         sys.exit("Usage: (CUDA_VISIBLE_DEVICES=[gpu], python test.py [version] [epoch])")
#     version = args[1]
#     epoch = args[2]
#     modelname = str(version) + "-" + str(epoch)

#     testset = FingerprintDataset(
#         mode="test",
#         levels="easy medium hard",
#         ratio=0.9,
#         imagetype="bmp",
#         damagetypes=["CR"],
#         transform=transforms.Compose([
#             transforms.Resize((200, 200)),
#             transforms.ToTensor()
#         ])
#     )
                                 
#     test_loader = torch.utils.data.DataLoader(
#         testset,
#         batch_size=32,
#         shuffle=False,
#         num_workers=1
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = torch.load(f"/home/kiss2023/workspace/Jayan/models/version{version}/fingerprintmodel{modelname}.pt")
#     model = model.to(device)

#     test(model, device, test_loader, modelname)

if __name__ == "__main__":
    main()