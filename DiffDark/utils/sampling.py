import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop
import numpy as np
import torchvision.transforms.functional as TF

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def generalized_steps(x, x_cond, x_r, x_h, seq, model, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            # et是模型预测的噪声ϵ
            et, m = model(torch.cat([x_cond, xt, x_r, x_h], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            # sigma
            # 这里面next指的是前一时刻也就是t-1时刻的x
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds

def generalized_steps_overlapping(x, x_cond, x_r, x_h, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        m2 = []

        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        my_mask = torch.ones_like(x_cond, device=x.device)
        my_mask *= 10.0
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et_output = torch.zeros_like(x_cond, device=x.device)
            m_output = torch.zeros_like(x_cond, device=x.device)
            
            if manual_batching:
                manual_batching_size = 64
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                x_r_patch = torch.cat([data_transform(crop(x_r, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                x_h_patch = torch.cat([data_transform(crop(x_h, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                for i in range(0, len(corners), manual_batching_size):
                    outputs, m = model(torch.cat([x_cond_patch[i:i+manual_batching_size], xt_patch[i:i+manual_batching_size],
                                              x_r_patch[i:i+manual_batching_size], x_h_patch[i:i+manual_batching_size]], dim=1), t)

                    for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
                        m_output[0, :, hi:hi + p_size, wi:wi + p_size] += m[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    x_cond_patch = data_transform(x_cond_patch)
                    x_r_patch = crop(x_r, hi, wi, p_size, p_size)
                    x_r_patch = data_transform(x_r_patch)
                    x_h_patch = crop(x_h, hi, wi, p_size, p_size)
                    x_h_patch = data_transform(x_h_patch)
                    et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch, x_r_patch,
                                                                                        x_h_patch], dim=1), t)

            et = torch.div(et_output, x_grid_mask)
            m1 = torch.div(m_output, x_grid_mask)
            m2.append(m1.to('cpu'))
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds, m2
