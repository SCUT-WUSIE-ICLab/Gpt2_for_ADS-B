import torch
import torch.nn as nn
from torch.nn import functional as F

'''
定义了几个损失函数类和一个用于获取相应损失函数模块的函数。
这些损失函数用于不同类型的机器学习任务（如分类、回归等）。

这些损失函数在不同的机器学习任务中非常有用，尤其是在处理时间序列数据或需要特殊损失计算方法的场景。
例如，MaskedMSELoss 在处理有缺失值的时间序列预测任务中非常实用，
而 NoFussCrossEntropyLoss 则在处理分类任务时提供了更大的灵活性。
'''

'''
一个函数，根据配置中的任务类型返回相应的损失函数模块。
支持的任务类型包括插值（imputation）、传导（transduction）、分类（classification）和回归（regression）。
'''
def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))

'''
一个函数，计算模型输出层的 L2 范数（平方和）。
通常用于正则化，帮助防止过拟合。
'''
def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))

'''
继承自 nn.CrossEntropyLoss，一个用于分类任务的交叉熵损失函数。
修改以接受更灵活的目标张量格式，并去除一些 PyTorch 原生交叉熵损失函数的限制。
'''
class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

'''
一个自定义的均方误差（MSE）损失函数，支持对输入数据应用掩码。
在计算损失时，可以忽略掩码为 0 的部分，只计算掩码为 1 的部分。
适用于处理具有缺失值的数据。
'''
class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
