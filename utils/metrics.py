import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def overall_loss_func(y_pred, y_true):
    """
    Mean Squared Error Loss.
    """
    return F.mse_loss(y_pred, y_true)

class MeanGradientError(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.sobel_x = sobel_x.repeat(3, 1, 1, 1) 
        self.sobel_y = sobel_y.repeat(3, 1, 1, 1) 

        self.register_buffer('sobel_x_filter', self.sobel_x)
        self.register_buffer('sobel_y_filter', self.sobel_y)

    def _sobel_edges(self, img):
        grad_x = F.conv2d(img, self.sobel_x_filter, padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, self.sobel_y_filter, padding=1, groups=img.shape[1])
        return grad_x, grad_y

    def forward(self, y_pred, y_true):
        gradients_true_x, gradients_true_y = self._sobel_edges(y_true)
        gradients_pred_x, gradients_pred_y = self._sobel_edges(y_pred)

        errors_x = torch.abs(gradients_true_x - gradients_pred_x)
        errors_y = torch.abs(gradients_true_y - gradients_pred_y)

        return torch.mean(errors_x + errors_y) / 2.0

# PSNR and SSIM from torchmetrics
def get_psnr_ssim_metrics(data_range=1.0):
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range)
    return psnr_metric, ssim_metric