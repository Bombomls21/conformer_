import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self, sample_weight=None, size_sum=True, scale=None):
        super(BCELoss, self).__init__()
        self.sample_weight = sample_weight # dùng để cân bằng các lớp
        self.size_sum = size_sum # sum hay mean loss
        self.hyper = 0.8 # # Giá trị scale loss
        self.smoothing = None # # Giá trị smoothing labels
        # [0, 1, 0, 1], alpha = 0.1 --> [0.1, 0.9, 0.1, 0.9] --> tránh quá khớp

    def forward(self, logits, targets):
        logits = logits

        if self.smoothing is not None: # Làm mịn targets
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Tạo targets mask để tính sample weight
        # [0.8, 0.3, 0.7, 0.1], threshold = 0,5 --> [1, 0, 1, 0]
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        # giúp balance giữa các lớp không cân bằng
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight) # Tính sample weight
            loss_m = (loss_m * sample_weight.cuda()) # Nhân với sample weight

        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return loss, loss_m

    
def ratio2weight(targets, ratio):
    # ratio = [1, 2], torch.tensor([1, 0, 1, 1, 0]
    ratio = torch.from_numpy(ratio).type_as(targets) # tensor([1, 2])

    pos_weights = targets * (1 - ratio) # Trọng số cho positive
    # pos_weights = (1 - 1) * torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)
    # tensor([0., 0., 0., 0., 0.])

    neg_weights = (1 - targets) * ratio # Trọng số cho negative
    # neg_weights = 2 * (1 - torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32))
    # tensor([2., 0., 2., 2., 0.])

    weights = torch.exp(neg_weights + pos_weights)
    # tensor([7.3891, 1.0000, 7.3891, 7.3891, 1.0000])

    weights[targets > 1] = 0.0 # Trọng số bằng 0 nếu targets > 1

    return weights