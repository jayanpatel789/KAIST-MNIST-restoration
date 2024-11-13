"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear=bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear=bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear=bilinear))
        self.up4 = (Up(128, 64, bilinear=bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        self.feature_maps = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4
        }
        return logits
    
    def get_feature_maps(self):
        """
        Method added to return the feature maps from the encoding process
        """
        return self.feature_maps
    
class UNetX(nn.Module):
    def __init__(self, n_channels, n_classes, feature_map_in, bilinear=False):
        """
        Addition of feature_map_in variable input. A list of feature maps from original
        U Net.
        """
        super(UNetX, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature_map_in = feature_map_in
        self.bilinear = bilinear
        factor = 2 if bilinear else 1 # Ignore state of bilinear in this model as we will not be using it

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, conv_channels=1536, bilinear=bilinear))
        self.up2 = (Up(512, 256 // factor, conv_channels=768, bilinear=bilinear))
        self.up3 = (Up(256, 128 // factor, conv_channels=384, bilinear=bilinear))
        self.up4 = (Up(128, 64, conv_channels=192, bilinear=bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4, self.feature_map_in["x4"])
        x = self.up2(x, x3, self.feature_map_in["x3"])
        x = self.up3(x, x2, self.feature_map_in["x2"])
        x = self.up4(x, x1, self.feature_map_in["x1"])
        logits = self.outc(x)
        self.feature_maps = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4
        }
        return logits
    
    def get_feature_maps(self):
        return self.feature_maps
    
def test():
    model0 = UNet(n_channels=1, n_classes=1)
    #feature_map = [torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)]
    model1 = UNetX(n_channels=1, n_classes=1, feature_map_in=None)
    input = torch.randn(1, 1, 180, 180) # Test input
    target = torch.randn(1, 1, 180, 180) # Test target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    with torch.no_grad():
        input, target = input.to(device), target.to(device)
        model0 = model0.to(device)
        
        inter = model0(input)
        feature_maps = model0.get_feature_maps()
        print(f"Intermediate shape: {inter.shape}")
        
        model1.feature_map_in = feature_maps
        inter = inter.to(device)
        model1 = model1.to(device)
        output = model1(inter)
        print(f"Output shape: {output.shape}")


    

if __name__ == "__main__":
    test()
