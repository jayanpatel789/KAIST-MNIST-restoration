"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

""" Full assembly of the parts to form the complete network """

from unet_parts import *


class DoubleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DoubleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024 // factor))
        self.up1a = (Up(1024, 512 // factor, bilinear=bilinear))
        self.up2a = (Up(512, 256 // factor, bilinear=bilinear))
        self.up3a = (Up(256, 128 // factor, bilinear=bilinear))
        self.up4a = (Up(128, 64, bilinear=bilinear))
        self.up1b = (Up(1024, 512 // factor, conv_channels=1536, bilinear=bilinear))
        self.up2b = (Up(512, 256 // factor, conv_channels=768, bilinear=bilinear))
        self.up3b = (Up(256, 128 // factor, conv_channels=384, bilinear=bilinear))
        self.up4b = (Up(128, 64, conv_channels=192, bilinear=bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # Encoding
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Store the feature maps from the first encoding in a dictionary
        self.original_feature_maps = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4
        }
        # Decoding
        x = self.up1a(x5, x4)
        x = self.up2a(x, x3)
        x = self.up3a(x, x2)
        x = self.up4a(x, x1)
        intermediate = self.outc(x)

        # Encoding stage 2
        x1 = self.inc(intermediate)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoding stage 2, with concatenation of first encoding feature maps
        x = self.up1b(x5, x4, self.original_feature_maps["x4"])
        x = self.up2b(x, x3, self.original_feature_maps["x3"])
        x = self.up3b(x, x2, self.original_feature_maps["x2"])
        x = self.up4b(x, x1, self.original_feature_maps["x1"])
        logits = self.outc(x)
        
        return logits
    
    def get_feature_maps(self):
        """
        Method added to return the feature maps from the encoding process
        """
        return self.feature_maps
    
def test():
    model = DoubleUNet(n_channels=1, n_classes=1)
    input = torch.randn(1, 1, 180, 180) # Test input
    target = torch.randn(1, 1, 180, 180) # Test target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input, target = input.to(device), target.to(device)
        model = model.to(device)
        
        output = model(input)
        print(f"Output shape: {output.shape}")
    

if __name__ == "__main__":
    test()
