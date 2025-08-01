# bc_model.py
import torch
import torch.nn as nn
from torchvision import models

NUM_BINS = 512      # x / y 各 512 个类别

class ResNetStateFusionTrans(nn.Module):
    """
    图像 + state → Transformer → 输出 (B, chunk, 2, 512) logits
    """
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size

        # ResNet-18（完全可训练）
        res = models.resnet18(weights="IMAGENET1K_V1")
        self.img_encoder = nn.Sequential(*list(res.children())[:-1])  # (B,512,1,1)
        self.img_proj    = nn.Linear(512, d_model)

        # state → d_model
        self.state_proj  = nn.Linear(2, d_model)

        # Transformer Encoder (seq_len=2)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            activation="gelu",
            batch_first=True,
            dropout=0.1,
        )
        self.trans_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # 输出 logits：chunk × 2 × 512
        self.head = nn.Linear(d_model, chunk_size * 2 * NUM_BINS)

    def forward(self, img, state):
        """
        img   : (B,3,H,W)
        state : (B,2)  — 已归一化 [-1,1]
        return: logits (B, chunk_size, 2, 512)
        """
        tok_img   = self.img_proj(self.img_encoder(img).flatten(1))
        tok_state = self.state_proj(state)
        seq       = torch.stack([tok_img, tok_state], dim=1)       # (B,2,256)
        enc       = self.trans_encoder(seq).mean(dim=1)            # (B,256)
        out       = self.head(enc)                                 # (B, chunk*2*512)
        return out.view(-1, self.chunk_size, 2, NUM_BINS)          # (B,chunk,2,512)
