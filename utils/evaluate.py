import torch
import logging

logger = logging.getLogger(__name__)

class ConfusionMatrixAllClass:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=self.device)

    def update(self, pred, target):
        """
        pred, target: [B, H, W] hoặc [H, W] – bất kỳ dtype nào
        → Fix 100% lỗi CUDA device-side assert
        """
        # BƯỚC 1: Chuyển về CPU + flatten + ép kiểu an toàn
        pred = pred.view(-1).cpu()
        target = target.view(-1).cpu()

        # BƯỚC 2: Loại bỏ hoàn toàn giá trị ngoài range (255, 4, -1, NaN...)
        valid_mask = (pred >= 0) & (pred < self.num_classes) & \
                    (target >= 0) & (target < self.num_classes)

        pred = pred[valid_mask]
        target = target[valid_mask]

        # Nếu không còn pixel hợp lệ → bỏ qua
        if pred.numel() == 0:
            return

        # BƯỚC 3: Chuyển sang long + đưa lên GPU (chỉ những giá trị đã sạch!)
        pred = pred.long().to(self.device)
        target = target.long().to(self.device)

        # BƯỚC 4: Tính chỉ số an toàn (không bao giờ ra ngoài n*n)
        n = self.num_classes
        idx = target * n + pred  # [N], giá trị từ 0 đến n*n-1 → 100% hợp lệ

        # BƯỚC 5: Dùng scatter_add_ – SIÊU NHANH và AN TOÀN
        self.mat += torch.bincount(idx, minlength=n * n).view(n, n)

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / (h.sum() + 1e-10)
        acc = torch.diag(h) / (h.sum(1) + 1e-10)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-10)
        dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0) + 1e-10)

        freq = h.sum(1) / (h.sum() + 1e-10)
        fw_iu = (freq[freq > 0] * iu[freq > 0]).sum()

        return (
            acc_global.item(),
            acc.cpu().numpy(),
            iu.cpu().numpy(),
            dice.cpu().numpy(),
            0.0,
            fw_iu.item()
        )

    def reset(self):
        self.mat.zero_()
        
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

        
