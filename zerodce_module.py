import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CSDN_Tem(nn.Module):
    """Depthwise Separable Convolution with Pointwise Convolution"""
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class ZeroDCEPlusPlus(nn.Module):
       
    def __init__(self, scale_factor=1.0, pretrained_path=None):
        super(ZeroDCEPlusPlus, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        # Zero-DCE++ DWC + p-shared architecture
        self.e_conv1 = CSDN_Tem(3, number_f) 
        self.e_conv2 = CSDN_Tem(number_f, number_f) 
        self.e_conv3 = CSDN_Tem(number_f, number_f) 
        self.e_conv4 = CSDN_Tem(number_f, number_f) 
        self.e_conv5 = CSDN_Tem(number_f*2, number_f) 
        self.e_conv6 = CSDN_Tem(number_f*2, number_f) 
        self.e_conv7 = CSDN_Tem(number_f*2, 3)
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, pretrained_path):
        """Load pretrained weights from Zero-DCE++ checkpoint"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Filter out incompatible keys
            model_dict = self.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def enhance(self, x, x_r):
        
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image
    
    def forward(self, x):
     
        # Multi-scale processing
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear')
        
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        
        # Generate enhancement curves
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        if self.scale_factor != 1:
            x_r = self.upsample(x_r)
        
        enhance_image = self.enhance(x, x_r)
        
        return enhance_image


class ZeroDCEPlusPlusYOLO(nn.Module):
       
    def __init__(self, scale_factor=1.0, pretrained_path=None, training_mode=True):
        super(ZeroDCEPlusPlusYOLO, self).__init__()
        
        self.zerodce = ZeroDCEPlusPlus(scale_factor=scale_factor, pretrained_path=pretrained_path)
        self.training_mode = training_mode
        
    def forward(self, x):
    
        if self.training_mode:
            # During training, we might want to add some noise or augmentation
            enhanced = self.zerodce(x)
            # Clamp to valid range
            enhanced = torch.clamp(enhanced, 0, 1)
            return enhanced
        else:
            # During inference, just enhance
            with torch.no_grad():
                enhanced = self.zerodce(x)
                enhanced = torch.clamp(enhanced, 0, 1)
                return enhanced
    
    def set_training_mode(self, training):
        """Set training mode"""
        self.training_mode = training
        self.zerodce.train(training)


# Factory function for easy integration
def create_zerodce_layer(scale_factor=1.0, pretrained_path=None, training_mode=True):
    return ZeroDCEPlusPlusYOLO(
        scale_factor=scale_factor,
        pretrained_path=pretrained_path,
        training_mode=training_mode
    )


if __name__ == "__main__":
    # Test the module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ZeroDCEPlusPlus(scale_factor=1.0).to(device)
    
    # Test with random input
    x = torch.randn(1, 3, 640, 640).to(device)
    
    # Forward pass
    with torch.no_grad():
        enhanced = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {enhanced.shape}")
    print(f"Output range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
