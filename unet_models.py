import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        input_channels = 3
        width = 64
        self.dconv_down1 = double_conv(input_channels, width)
        self.dconv_down2 = double_conv(width, width * 2)
        self.dconv_down3 = double_conv(width * 2, width * 4)
        self.dconv_down4 = double_conv(width * 4, width * 8)

        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_up3 = double_conv(width * (4 + 8), width * 4)
        self.dconv_up2 = double_conv(width * (2 + 4), width * 2)
        self.dconv_up1 = double_conv(width * (2 + 1), width)
        
        self.conv_last = nn.Conv2d(width, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        
        return out

    def print_layer(self, layer):
        for name, param in layer.named_parameters():
            print("name: {}\nparameters: {}".format(name, list(param.detach())))

    def print(self):
        print(self.__class__.__name__)
        encoder_list = [self.dconv_down1, self.dconv_down2, self.dconv_down3, self.dconv_down4]
        for i, layer in enumerate(encoder_list):
            print('Encoder layer {}: {}'.format(i, layer))
            self.print_layer(layer)
        decoder_list = [self.dconv_up3, self.dconv_up2, self.dconv_up1, self.conv_last]
        for i, layer in enumerate(decoder_list):
            print('Decoder layer {}: {}'.format(i, layer))
            self.print_layer(layer)
            break


class uunet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        width = 64
        input_channels = 3
        second_input_dimsize = input_channels + 1
        self.dconv_down1 = double_conv(input_channels, width)
        self.dconv_down2 = double_conv(width, 2 * width)
        self.dconv_down3 = double_conv(2 * width, 4 * width)
        self.dconv_down4 = double_conv(4 * width, 8 * width)

        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv((4 + 8) * width, 4 * width)
        self.dconv_up2 = double_conv((2 + 4) * width, 2 * width)
        self.dconv_up1 = double_conv((2 + 1) * width, width)

        self.dconv_down1_2 = double_conv(second_input_dimsize, width)
        self.dconv_down2_2 = double_conv((1 + 2) * width, 2 * width)
        self.dconv_down3_2 = double_conv((2 + 4) * width, 4 * width)
        self.dconv_down4_2 = double_conv((4 + 8) * width, 8 * width)

        self.dconv_up3_2 = double_conv((4 + 8) * width, 4 * width)
        self.dconv_up2_2 = double_conv((2 + 4) * width, 2 * width)
        self.dconv_up1_2 = double_conv((2 + 1) * width, width)

        self.conv_bin_last = nn.Conv2d(width, 1, 1)
        self.conv_last = nn.Conv2d(width, n_class, 1)

    def forward(self, x):
        input = x
        conv1 = self.dconv_down1(x)

        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)

        x = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        dconv3 = self.dconv_up3(x)

        x = F.interpolate(dconv3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)
        dconv2 = self.dconv_up2(x)

        x = F.interpolate(dconv2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out_bin = self.conv_bin_last(x)
        out_bin_act = self.sigmoid(out_bin)

        img_bin = torch.cat([input[:, :, :, :], out_bin_act], dim=1)

        conv1 = self.dconv_down1_2(img_bin)

        x = self.maxpool(conv1)

        x = torch.cat([x, dconv2], dim=1)
        conv2 = self.dconv_down2_2(x)

        x = self.maxpool(conv2)
        x = torch.cat([x, dconv3], dim=1)
        conv3 = self.dconv_down3_2(x)

        x = self.maxpool(conv3)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_down4_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1_2(x)

        out_mult = self.conv_last(x)

        return out_bin, out_mult


class wuunet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        width = 64
        input_channels = 3
        second_input_dimsize = input_channels + 1
        self.dconv_down1 = double_conv(input_channels, width)
        self.dconv_down2 = double_conv(width, 2 * width)
        self.dconv_down3 = double_conv(2 * width, 4 * width)
        self.dconv_down4 = double_conv(4 * width, 8 * width)

        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv((4 + 8) * width, 4 * width)
        self.dconv_up2 = double_conv((2 + 4) * width, 2 * width)
        self.dconv_up1 = double_conv((2 + 1) * width, width)

        self.dconv_down1_2 = double_conv(second_input_dimsize, width)
        self.dconv_down2_2 = double_conv((1 + 2 + 2) * width, 2 * width)
        self.dconv_down3_2 = double_conv((2 + 4 + 4) * width, 4 * width)
        self.dconv_down4_2 = double_conv((4 + 8) * width, 8 * width)

        self.dconv_up3_2 = double_conv((4 + 8 + 8) * width, 4 * width)
        self.dconv_up2_2 = double_conv((2 + 4 + 4) * width, 2 * width)
        self.dconv_up1_2 = double_conv((2 + 1) * width + second_input_dimsize, width)

        self.conv_bin_last = nn.Conv2d(width, 1, 1)
        self.conv_last = nn.Conv2d(width, n_class, 1)

    def forward(self, x):
        input = x
        conv1 = self.dconv_down1(x)

        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)

        x = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        dconv3 = self.dconv_up3(x)

        x = F.interpolate(dconv3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)
        dconv2 = self.dconv_up2(x)

        x = F.interpolate(dconv2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out_bin = self.conv_bin_last(x)
        out_bin_act = self.sigmoid(out_bin)

        img_bin = torch.cat([input[:, :, :, :], out_bin_act], dim=1)

        conv1_1 = self.dconv_down1_2(img_bin)

        x = self.maxpool(conv1_1)

        x = torch.cat([x, conv2, dconv2], dim=1)
        conv2_1 = self.dconv_down2_2(x)

        x = self.maxpool(conv2_1)
        x = torch.cat([x, conv3, dconv3], dim=1)
        conv3_1 = self.dconv_down3_2(x)

        x = self.maxpool(conv3_1)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_down4_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3, dconv3, conv3_1], dim=1)
        x = self.dconv_up3_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2, dconv2, conv2_1], dim=1)
        x = self.dconv_up2_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1, img_bin], dim=1)
        x = self.dconv_up1_2(x)

        out_mult = self.conv_last(x)

        return out_bin, out_mult
