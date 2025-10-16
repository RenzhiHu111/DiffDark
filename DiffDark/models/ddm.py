import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
import math
from diffdark_ssim import *
from torch.cuda.amp import autocast, GradScaler

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def tv_loss(input, output):
    I = torch.mean(input, dim=1, keepdim=True)
    L = torch.log(I + 0.0001)
    I1 = torch.mean(output, dim=1, keepdim=True)
    L1 = torch.log(I1 + 0.0001)
    dx = L[:, :, :-1, :-1] - L[:, :, :-1, 1:]
    dy = L[:, :, :-1, :-1] - L[:, :, 1:, :-1]

    alpha = torch.tensor(1.2)
    lamda = torch.tensor(1.5)
    dx = lamda / (torch.abs(dx) ** alpha + torch.tensor(0.0001))
    dy = lamda / (torch.abs(dy) ** alpha + torch.tensor(0.0001))

    x_loss = dx * ((L1[:, :, :-1, :-1] - L1[:, :, :-1, 1:]) ** 2)
    y_loss = dy * ((L1[:, :, :-1, :-1] - L1[:, :, 1:, :-1]) ** 2)
    tvloss = torch.mean(x_loss + y_loss) / 2.0
    return tvloss

def noise_estimation_loss(model, x0, t, e, b):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:6, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output, m = model(torch.cat([x0[:, :3, :, :], x, x0[:, 6:9, :, :], x0[:, 9:12, :, :]], dim=1), t.float())
    noise_loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    criterion1 = nn.MSELoss()
    at_loss = criterion1(x0[:,  12:15, :, :], m)

    x_0 = (x - output * (1.0 - a).sqrt()) / a.sqrt()

    criterion = nn.MSELoss()
    loss3 = criterion(x_0, x0[:, 3:6, :, :])

    ssim_loss = SSIM(window_size=11)
    loss2 = 1 - ssim_loss(x_0, x0[:, 3:6, :, :])

    scale_color = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss1 = torch.mean(-1 * scale_color(x_0, x0[:, 3:6, :, :]))

    sum = noise_loss + 10 * loss1 + 10 * loss2 + 10 * loss3 + at_loss

    return noise_loss, loss1, loss2, loss3, at_loss, sum


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)

        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:6, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                noise_loss, loss1, loss2, loss3, at_loss, all_loss = noise_estimation_loss(self.model, x, t, e, b)
                loss = all_loss

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i + 1)}")
                    print(f"noise_loss: {noise_loss.item()}")
                    print(f"loss1: {loss1.item()}")
                    print(f"loss2: {loss2.item()}")
                    print(f"loss3: {loss3.item()}")
                    print(f"at_loss: {at_loss.item()}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_ddpm'))

    def sample_image(self, x_cond, x, x_r, x_h, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        start_value = 999  # 起始值
        num_samples = 25  # 采样数量
        samples = []
        start_tolerance = 1
        tolerance = (start_value / (num_samples - start_tolerance) - start_tolerance) / (num_samples - start_tolerance)

        for i in range(num_samples):
            samples.append(start_value - i * start_tolerance)
            start_tolerance += tolerance
        samples = [math.floor(value) for value in samples]
        samples = [0 if value < 0 else value for value in samples]
        # print(samples)
        samples.reverse()
        seq = samples

        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, x_r, x_h, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, x_r, x_h, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x_r = x[:, 6:9, :, :].to(self.device)
            x_r = data_transform(x_r)
            x_h = x[:, 9:12, :, :].to(self.device)
            x_h = data_transform(x_h)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x, x_r, x_h)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)
            x_r = inverse_data_transform(x_r)
            x_h = inverse_data_transform(x_h)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
                utils.logging.save_image(x_r[i], os.path.join(image_folder, str(step), f"{i}_reflection.png"))
                utils.logging.save_image(x_h[i], os.path.join(image_folder, str(step), f"{i}_he.png"))
