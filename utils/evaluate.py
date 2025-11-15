import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class ConfusionMatrixAllClass:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat1 = torch.zeros((num_classes, num_classes), dtype=torch.int64, device='cuda')
        self.mat2 = torch.zeros((num_classes, num_classes), dtype=torch.int64, device='cuda')

    def update(self, a, b):
        a = a.long()
        b = b.long()
        if a.device != self.mat1.device:
            logger.warning(f"Device mismatch: a on {a.device}, mat1 on {self.mat1.device}. Moving a and b to {self.mat1.device}.")
            a = a.to(self.mat1.device)
            b = b.to(self.mat1.device)
        if a.device != b.device:
            logger.warning(f"Device mismatch: a on {a.device}, b on {b.device}. Moving b to {a.device}.")
            b = b.to(a.device)
        if a.min() < 0 or a.max() >= self.num_classes or b.min() < 0 or b.max() >= self.num_classes:
            logger.warning(f"Invalid class indices: a min={a.min()}, max={a.max()}, b min={b.min()}, max={b.max()}")
            a = torch.clamp(a, 0, self.num_classes - 1)
            b = torch.clamp(b, 0, self.num_classes - 1)
        n = self.num_classes
        self.mat1 += torch.bincount(a.view(-1) * n + b.view(-1), minlength=n * n).reshape(n, n).to(self.mat1.device)
        self.mat2 += torch.bincount(b.view(-1) * n + a.view(-1), minlength=n * n).reshape(n, n).to(self.mat2.device)

    def compute(self):
        h = self.mat1.float()
        acc_global = torch.nan_to_num(torch.diag(h).sum() / h.sum(), nan=0.0)
        acc = torch.nan_to_num(torch.diag(h) / h.sum(1), nan=0.0)
        iu = torch.nan_to_num(torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h)), nan=0.0)
        dice_per_class = torch.nan_to_num(2 * torch.diag(h) / (h.sum(1) + h.sum(0)), nan=0.0)
        dice_bg_fg = (dice_per_class[0] + dice_per_class[-1]) / 2
        freq = h.sum(1) / h.sum()
        fw_iu = (freq[freq > 0] * iu[freq > 0]).sum()
        return (
            acc_global.item(),
            acc.cpu().numpy(),
            iu.cpu().numpy(),
            dice_per_class.cpu().numpy(),
            dice_bg_fg.item(),
            fw_iu.item(),
        )
        
    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat1)
        torch.distributed.all_reduce(self.mat2)

    def __str__(self):
        acc_global, acc, iu, dice, fg_bg_dice, fw_iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}\n'
            'dice: {}\n'
            'mean dice: {}\n'
            'fg_bg_dice: {}\n'
            'mean_fg_bg: {}\n'
            'fw_iu: {:.1f}').format(
                acc_global * 100,
                ['{:.1f}'.format(i * 100) for i in acc],
                ['{:.1f}'.format(i * 100) for i in iu],
                iu[:-1].mean() * 100,
                ['{:.1f}'.format(i * 100) for i in dice],
                dice[:-1].mean() * 100,
                ['{:.1f}'.format(i * 100) for i in fg_bg_dice],
                fg_bg_dice.mean() * 100,
                fw_iu * 100
            )
            