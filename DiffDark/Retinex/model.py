import os
import time
import random
import math

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from deform import DeformConv2d

from base_layers import *

from Discriminator import *

Sobel = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
Robert = np.array([[0, 0],
                   [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

def gradient_no_abs(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=2, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=48):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction_ratio

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (self.reduced_channels**(-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7], final_out_channels=None):
        super(MultiScaleBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2, padding_mode='replicate')
            for k in kernels
        ])
        self.final_out_channels = final_out_channels if final_out_channels is not None else out_channels * len(kernels)
        self.adjust_conv = nn.Conv2d(out_channels * len(kernels), self.final_out_channels, kernel_size=1)
    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        combined_features = torch.cat(features, dim=1)
        adjusted_features = self.adjust_conv(combined_features)
        return adjusted_features

# DecomNet
class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode='replicate')
        self.multi_scale_block = MultiScaleBlock(channel, channel, kernels=[3, 5, 7], final_out_channels=channel)
        self.net1_conv1 = ResnetBlock(in_channels=channel, out_channels=channel)
        self.net1_conv2 = ResnetBlock(in_channels=channel, out_channels=channel)
        self.net1_conv3 = ECAAttention()
        self.net1_conv4 = ResnetBlock(in_channels=channel, out_channels=channel)
        self.net1_conv5 = ResnetBlock(in_channels=channel, out_channels=channel)
        self.net1_conv6 = DeformConv2d(channel, channel)
        self.net1_conv8 = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        feats0_1 = self.multi_scale_block(feats0)
        feats1 = self.net1_conv1(feats0_1)
        feats2 = self.net1_conv2(feats1)
        feats3 = self.net1_conv3(feats2)
        feats4 = self.net1_conv4(feats3)
        feats5 = self.net1_conv5(feats4)
        feats6 = self.net1_conv6(feats5)
        outs = self.net1_conv8(feats6)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

# Enhance-Net
class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        # self.quzao=Restormer()

        filters = 32
        self.conv1_1 = Conv2D(4, filters)
        self.conv1_2 = Conv2D(filters, filters)
        self.pool1 = MaxPooling2D()

        self.conv2_1 = Conv2D(filters, filters * 2)
        self.conv2_2 = Conv2D(filters * 2, filters * 2)
        self.pool2 = MaxPooling2D()

        self.conv3_1 = Conv2D(filters * 2, filters * 4)
        self.conv3_2 = Conv2D(filters * 4, filters * 4)
        self.pool3 = MaxPooling2D()

        self.conv5_1 = Conv2D(filters * 4, filters * 8)
        self.conv5_2 = Conv2D(filters * 8, filters * 8)
        # self.my1 = MyAttention(in_channels=filters * 8)

        self.upv6 = ConvTranspose2D(filters * 8, filters * 4)
        self.concat6 = Concat()
        self.conv6_1 = Conv2D(filters * 8, filters * 4)
        self.conv6_2 = Conv2D(filters * 4, filters * 4)

        self.upv7 = ConvTranspose2D(filters * 4, filters * 2)
        self.concat7 = Concat()
        self.conv7_1 = Conv2D(filters * 4, filters * 2)
        self.conv7_2 = Conv2D(filters * 2, filters * 2)

        self.upv8 = ConvTranspose2D(filters * 2, filters)
        self.concat8 = Concat()
        self.conv8_1 = Conv2D(filters * 2, filters)
        self.conv8_2 = Conv2D(filters, filters)

        self.conv9_1 = nn.Conv2d(filters, 3, kernel_size=1, stride=1)
        self.out1 = nn.Sigmoid()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')

        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)


    def forward(self, input_L, input_R):

        input_img = torch.cat((input_R, input_L), dim=1)

        conv1 = self.conv1_1(input_img)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool1(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool1(conv3)

        conv5 = self.conv5_1(pool3)
        conv5 = self.my1(conv5,1,1)

        up6 = self.upv6(conv5)
        up6 = self.concat6(conv3, up6)
        conv6 = self.conv6_1(up6)
        conv6 = self.conv6_2(conv6)

        up7 = self.upv7(conv6)
        up7 = self.concat7(conv2, up7)
        conv7 = self.conv7_1(up7)
        conv7 = self.conv7_2(conv7)

        up8 = self.upv8(conv7)
        up8 = self.concat8(conv1, up8)
        conv8 = self.conv8_1(up8)
        conv8 = self.conv8_2(conv8)

        outx = self.conv9_1(conv8)
        out = self.out1(outx)

        out0      = self.net2_conv0_1(input_img)
        out1      = self.relu(self.net2_conv1_1(out0))
        out2      = self.relu(self.net2_conv1_2(out1))
        out3      = self.relu(self.net2_conv1_3(out2))

        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))

        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs= F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs= F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output    = self.net2_output(feats_fus)
        output1 = torch.cat((output, output, output), dim=1)
        output2 = output1 * out

        return output, output1, out, output2

class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.ssim_loss = SSIM()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().cuda()

        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()
        self.discriminator = Discriminator()

    def forward(self, input_low, input_high, hrz=True):
        # Forward DecompNet
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        input_high = Variable(torch.FloatTensor(torch.from_numpy(input_high))).cuda()
        R_low, I_low = self.DecomNet(input_low)

        if hrz:
            R_high, I_high = self.DecomNet(input_high)
            # Forward RelightNet
            I_delta, I_delta_3, out, output = self.RelightNet(I_low, R_low)

            # Other variables
            I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
            I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)

            # --------------------------------------------------------------------------------------------------------------
            # train discriminator
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)

            self.label_cover = torch.full((16, 1), 1, dtype=torch.float, device=device)
            self.label_encoded = torch.full((16, 1), 0, dtype=torch.float, device=device)

            images = input_high
            encoded_images = output

            # RAW : target label for image should be "cover"(1)
            d_label_cover = self.discriminator(images)
            self.d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

            # GAN : target label for encoded image should be "encoded"(0)
            d_label_encoded = self.discriminator(encoded_images.detach())
            self.d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

            g_label_decoded = self.discriminator(encoded_images)
            self.g_loss_on_discriminator = self.criterion_BCE(g_label_decoded,
                                                              self.label_cover[:g_label_decoded.shape[0]])
            # --------------------------------------------------------------------------------------------------------------

            # Compute losses
            self.recon_loss_low = F.l1_loss(R_low * I_low_3, input_low)
            self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)

            self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low)
            self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)

            self.equal_R_loss = F.l1_loss(R_low, R_high.detach())
            # self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)  # 这边是L1损失,这个是没去噪的反射图（用错了）

            self.Ismooth_loss_low = self.smooth(I_low, R_low)
            self.Ismooth_loss_high = self.smooth(I_high, R_high)
            self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

            self.smooth_I = F.l1_loss(I_delta_3, I_high_3)
            self.SSIM_I = 1 - self.ssim_loss(I_delta_3, I_high_3)

            self.relight_loss1 = F.l1_loss(out * I_delta_3, input_high)
            self.de_loss = F.mse_loss(output, input_high)

            # self.psnr1 = 20 * math.log10(255 / math.sqrt(self.de_loss))
            self.psnr1 = 20 * math.log10(1 / math.sqrt(self.de_loss))

            self.loss_recon = F.l1_loss(out, R_high)
            self.SSIM_R = 1 - self.ssim_loss(out, R_high)
            x_loss = F.mse_loss(gradient_no_abs(out, 'x'), gradient_no_abs(R_high, 'x'))
            y_loss = F.mse_loss(gradient_no_abs(out, 'y'), gradient_no_abs(R_high, 'y'))
            self.grad_loss_R = x_loss + y_loss

            self.loss_Decom = self.recon_loss_low + \
                              self.recon_loss_high + \
                              0.001 * self.recon_loss_mutal_low + \
                              0.001 * self.recon_loss_mutal_high + \
                              0.1 * self.Ismooth_loss_low + \
                              0.1 * self.Ismooth_loss_high + \
                              0.01 * self.equal_R_loss

            self.loss_Relight = self.relight_loss1 + \
                                self.de_loss + \
                                self.smooth_I + \
                                self.SSIM_I + \
                                3 * self.Ismooth_loss_delta + \
                                self.loss_recon + \
                                self.SSIM_R + \
                                self.grad_loss_R + \
                                0.0001 * self.g_loss_on_discriminator

            self.output_R_low = R_low.detach().cpu()
            self.output_I_low = I_low_3.detach().cpu()
            self.output_I_delta = I_delta_3.detach().cpu()
            self.output_S = output.detach().cpu()
            self.out = out.detach().cpu()
        else:
            R_low, I_low = self.DecomNet(input_low)
            I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
            self.output_R_low = R_low.detach().cpu()
            self.output_I_low = I_low_3.detach().cpu()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img = Image.open(eval_low_data_names[idx])
            eval_low_img = np.array(eval_low_img, dtype="float32") / 255.0
            eval_low_img = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval, hrz=False)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image = np.concatenate([input, result_1, result_2], axis=2)
                # cat_image = result_1
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                result_5 = self.out
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                result_5 = np.squeeze(result_5)
                cat_image = np.concatenate([input, result_1, result_2, result_5, result_3, result_4], axis=2)
                # cat_image = np.concatenate([result_4], axis=2)
                # cat_image = np.concatenate([result_5], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                                    (train_phase, idx + 1, epoch_num))
            im.save(filepath[:-4] + '.jpg')

    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name = save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(), save_name)

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts) > 0:
                load_ckpt = load_ckpts[-1]
                global_step = int(load_ckpt[:-4])
                ckpt_dict = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0

    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom = optim.Adam(self.DecomNet.parameters(),
                                         lr=lr[0], betas=(0.9, 0.999))
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(),
                                            lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        self.train_phase = train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
              (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        self.sum=0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            self.sum = self.sum/30
            self.sum = 0
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.opt_discriminator.param_groups:
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                batch_input_high = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32') / 255.0
                    train_high_img = Image.open(train_high_data_names[image_id])
                    train_high_img = np.array(train_high_img, dtype='float32') / 255.0
                    # Take random crops
                    h, w, _ = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_high_img = train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_high_img = np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_high_img = np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_high_img = np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    train_high_img = np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    batch_input_high[patch_id, :, :, :] = train_high_img
                    message = torch.Tensor(np.random.choice([0, 1], (batch_input_low.shape[0], 30))).cuda()
                    self.input_low = batch_input_low
                    self.input_high = batch_input_high
                    self.message = message

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)

                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low, self.input_high, hrz=True)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "Relight":
                    self.opt_discriminator.zero_grad()
                    self.train_op_Relight.zero_grad()

                    self.d_cover_loss.backward()
                    self.d_encoded_loss.backward()

                    self.loss_Relight.backward()
                    self.opt_discriminator.step()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                    print('relight_loss1: %.6f' % float(self.relight_loss1))
                    print('de_loss: %.6f' % float(self.de_loss))

                    print('smooth_I: %.6f' % float(self.smooth_I))
                    print('SSIM_I: %.6f' % float(self.SSIM_I))
                    print('Ismooth_loss_delta: %.6f' % float(self.Ismooth_loss_delta))

                    print('loss_recon: %.6f' % float(self.loss_recon))
                    print('SSIM_R: %.6f' % float(self.SSIM_R))
                    print('grad_loss_R: %.6f' % float(self.grad_loss_R))

                    print('d_cover_loss: %.6f' % float(self.d_cover_loss))
                    print('d_encoded_loss: %.6f' % float(self.d_encoded_loss))
                    print('g_loss_on_discriminator: %.6f' % float(self.g_loss_on_discriminator))

                    print('psnr：%.6f' % float(self.psnr1))

                    self.sum += loss
                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)

    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_dir):

        # Load the network with a pre-trained checkpoint
        self.train_phase = 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        # self.train_phase = 'Relight'
        # load_model_status, _ = self.load(ckpt_dir)
        # if load_model_status:
        #     print(self.train_phase, ": Model restore success!")
        # else:
        #     print("No pretrained model to restore!")
        #     raise Exception

        # Set this switch to True to also save the reflectance and shading maps

        # Predict for the my images
        i = 1
        for idx in range(len(test_low_data_names)):
            test_img_path = test_low_data_names[idx]
            test_img_name = test_img_path.split('\\')[-1]
            print('Processing ', test_img_name)
            test_low_img = Image.open(test_img_path)
            test_low_img = np.array(test_low_img, dtype="float32") / 255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)
            self.forward(input_low_test, input_low_test,hrz=False)

            result_1 = self.output_R_low
            result_2 = self.output_I_low
            # result_3 = self.output_I_delta
            # result_4 = self.output_S
            # result_5 = self.out
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            # result_3 = np.squeeze(result_3)
            # result_4 = np.squeeze(result_4)
            # result_5 = np.squeeze(result_5)
            # if save_R_L:
            #     cat_image = np.concatenate([input, result_1, result_2, result_5, result_3, result_4], axis=2)
            # else:
            #     cat_image = np.concatenate([input, result_4], axis=2)
            cat_image = np.concatenate([result_1],axis=2)
            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = res_dir + '/' + test_img_name
            # filepath = res_dir + test_img_name
            im.save(filepath[:-4] + '.png')