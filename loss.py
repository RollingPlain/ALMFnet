from tools import Sobelxy
import torch


def FusionLoss(fus, mri, oth, mri_mask, oth_mask, seg, l1_loss, mse_loss, ssim_loss = None):
    # Pixel Loss
    loss_pixel = mse_loss(fus * mri_mask, mri * mri_mask) + mse_loss(fus * oth_mask, oth * oth_mask)

    # Gradient Loss
    sobelconv = Sobelxy()
    grad_max = torch.max(sobelconv(mri),sobelconv(oth))
    grad_fus = sobelconv(fus)
    loss_grad = l1_loss(grad_fus, grad_max)

    # Segment Loss
    loss_seg = l1_loss(fus * seg, mri * seg)
    seg = torch.where(seg <= 0, torch.ones_like(seg) * 0.25, torch.ones_like(seg) * 0.75)

    loss_total = 1.2*loss_pixel + 0.5 * loss_grad + 0.3 * loss_seg

    return loss_total