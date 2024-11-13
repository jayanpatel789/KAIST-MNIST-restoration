import os
from PIL import Image
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
from math import floor
from random import shuffle

default_transform = transforms.Compose([
            transforms.Resize((103, 96)),
            transforms.ToTensor()
        ])

class FingerprintDataset(Dataset):
    """
    Dataset class to be used by PyTorch dataloader for the return of image tensors from the SOCOFing dataset
    along with other relevant information.

    Inputs:
        mode: Either "train" or "test"
        levels: Which levels of testing should be included in the dataset: "easy", "medium" and/or "hard"
        ratio: The split of the whole dataset between the number of training samples and number of testing samples. Train/Whole dataset
        imagetype: "bmp" or "png"
        damagetype: Which damage types should be included in the dataset. "Obl", "ZCut" and/or "CR"
        transform: Transformations to perform on the image file before it is returned.
    """
    def __init__(self, mode, levels, ratio, imagetype,
                 damagetypes=["Obl", "Zcut", "CR"], transform=default_transform):
        # Initialise values and directories
        self.mode = mode
        self.imagetype = imagetype
        self.altered_root = f"/home/kiss2023/workspace/SOCOFing{imagetype.upper()}/Altered"
        self.real_root = f"/home/kiss2023/workspace/SOCOFing{imagetype.upper()}/Real"
        easy_root = "Altered-Easy"
        med_root = "Altered-Medium"
        hard_root = "Altered-Hard"

        # Select directories for chosen damage levels to be selected
        roots = []
        if "easy" in levels:
            roots.append(easy_root)
        if "medium" in levels:
            roots.append(med_root)
        if "hard" in levels:
            roots.append(hard_root)
        
        # Array of sizes of each damaged difficulty level
        total_sizes = []
        for root in roots:
            path = os.path.join(self.altered_root, root)
            size = len(os.listdir(path))
            total_sizes.append(size)

        # From the total number of altered images, calculate the number of images that will be used
        # for training using the train ratio. If test is selected, this is handled using the train sizes.
        # This allows for even split of dataset, irrespective of difficulties that are selected.
        train_sizes = []
        for size in total_sizes:
            set_size = floor(size*ratio)
            train_sizes.append(set_size)
        
        # Create a list of all the image paths that will be used as the dataset that the dataloader iterates over.
        self.imagelist = []
        # Loop for each damage level
        for i, root in enumerate(roots):
            path = os.path.join(self.altered_root, root)
            # Store list of all chosen image file names in images variable. images = [a.bmp, b.bmp, c.bmp, ...]
            if mode == "train":
                images = os.listdir(path)[:train_sizes[i]]
            elif mode == "test":
                images = os.listdir(path)[train_sizes[i]:]
            else:
                raise Exception("mode parameter is incorrect")
            
            # Iterate through list, adding each image name to the level directory. This helps to preserve the difficulty
            # in the name of image within self.imagelist
            
            # for j in range(len(images)):
            #     images[j] = os.path.join(root, images[j])

            for image in images:
                counter = 0
                # Check that a selected damagetype is present in the image name. If not, do not add image name to self.imagelist
                # This may skew the balance of the data split as no ratios are considered but the implementation was done later
                # for purpose of general investigation.
                for damagetype in damagetypes:
                    if damagetype.lower() in image.lower():
                        counter+=1
                if counter > 0:
                    image = os.path.join(root, image) # Join damage level root to image name
                    self.imagelist.append(image)
        
        shuffle(self.imagelist)

        self.transform = transform

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        keyword = 'finger'
        imagepath = self.imagelist[idx]
        if "Easy" in imagepath:
            level = "easy"
        elif "Medium" in imagepath:
            level = "medium"
        elif "Hard" in imagepath:
            level = "hard"
        altered_path = os.path.join(self.altered_root, imagepath)
        split = imagepath.split(keyword)
        subject = split[0] + keyword
        if "CR" in split[1]:
            damage = "CR"
        elif "Obl" in split[1]:
            damage = "Obl"
        elif "Zcut" in split[1]:
            damage = "Zcut"
        split = subject.split('/')
        subject = split[1]
        if self.imagetype == "bmp":
            real_name = subject + ".BMP"
        elif self.imagetype == "png":
            real_name = subject + ".png"
        real_path = os.path.join(self.real_root, real_name)
        paths = [altered_path, real_path]

        tensors = []
        for path in paths:
            image = Image.open(path)
            image = image.convert('L')
            tensor = self.transform(image)
            image.close()
            _, H, W = tensor.shape
            tensor = transforms.CenterCrop([H-20, W-20])(tensor)
            tensors.append(tensor)

        output = {
            "input": tensors[0],
            "inputpath": paths[0],
            "target": tensors[1],
            "outputpath": paths[1],
            "subjects": subject,
            "damages": damage,
            "levels": level
        }

        return output
    

def test():
    dataset = FingerprintDataset(
        mode="test",
        levels="hard",
        ratio=0.9,
        imagetype="bmp",
        damagetypes=["CR"],
        transform=transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor()
        ])
    )

    dataloader = Data.DataLoader(
        dataset=dataset,
        batch_size=5,
        shuffle=True
    )

    i = 1
    for batch in dataloader:
        i += 1
        if i >= 5:
            break
        inputs = batch["input"]
        targets = batch["target"]
        subjects = batch["subjects"]
        damages = batch["damages"]
        levels = batch["levels"]

        # print('input shape : ', inputs.shape)
        # print('target shape : ', targets.shape)
        # if inputs[0].shape != targets[0].shape:
        #     print(f"Shapes not equal!!. Subject, damage: {subjects[0]}, {damages[0]}")
        #     print(f"Input shape: {inputs[0].shape}, Target shape: {targets[0].shape}")
        # imageresult = torch.stack([inputs[0], targets[0]], dim=0)
        # filepath = f"/home/kiss2023/workspace/Jayan/U-Net3/datatest/data2test-{subjects[0]}-stack.png"
        #save_image(imageresult, filepath)

        #print("Input: ", inputs.shape)
        #print("Target: ", targets.shape)
        #print("Subjects: ", subjects)
        print("Damages: ", damages)
        #print("Levels: ", levels)
        # print('filepath : ', filepath)


if __name__ == "__main__":
    test()