"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)  # contiguous()为了保证内存的连续性，view（-1，1）第一个dim自动推断，第二个dim为1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # eq()函数的作用是判读对比双方是否equal，相等就是1，不相等就是0。样本之间是否为相同标签
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 获取对比视图的数量
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 把2个视图的向量特征拼接在一起
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count  # 每个图像都有2个锚节点
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits。计算相似度使用inner product
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 在第二个dim取最大值，并返回最大值和其索引
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # repeat的作用是,沿着指定的维度复制tensor
        # mask-out self-contrast cases 对角线置0，不计算自身的对比损失
        logits_mask = torch.scatter(  # 将某个Tensor中的特定索引的值,赋值(scatter)给另一个Tensor指定的索引。
            torch.ones_like(mask),  # 目标Tensor,被赋值
            1,  # 沿着哪个维度赋值
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  # 需要赋值的索引
            0  # 要赋的值
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 所有样本的对比损失，包括 正样本对 与 负样本对

        # compute mean of log-likelihood over positive正样本对
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
