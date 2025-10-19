import torch
import torch.nn as nn
from .ecapa_tdnn import ECAPA_TDNN

class TPredicter(nn.Module):
    def __init__(self, C):
        super(TPredicter, self).__init__()
        self.ecapa_tdnn = ECAPA_TDNN(C=C)
        self.output_activ = nn.Sigmoid()
        self.output_layer = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.SiLU(),
            nn.Linear(192, 1),
        )

    def forward(self, mixture, enrollment, aug=False):
        """
        Args:
            mixture (torch.Tensor): Noisy input tensor of shape (batch_size, time_steps)
            enrollment (torch.Tensor): Enrollment tensor of shape (batch_size, time_steps)
        
        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, C)
        """
        # Pass through ECAPA-TDNN
        enrollment_feat = self.ecapa_tdnn(enrollment, aug)
        mixture_feat = self.ecapa_tdnn(mixture, aug)
        sqrt_d = enrollment_feat.shape[1] ** 0.5
        enrollment_feat = enrollment_feat / sqrt_d
        mixture_feat = mixture_feat / sqrt_d

        # simularity = torch.einsum('bd,bd->b', enrollment_feat, mixture_feat)
        # simularity = self.cos_sim(enrollment_feat, mixture_feat)
        simularity = torch.cat([enrollment_feat, mixture_feat], dim=-1)
        simularity = self.output_layer(simularity).squeeze(-1)
        t = self.output_activ(simularity)
        
        return t