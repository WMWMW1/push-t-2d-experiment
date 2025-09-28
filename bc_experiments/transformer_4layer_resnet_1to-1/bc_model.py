# bc_model.py
import torch, torch.nn as nn
from torchvision import models

class ResNetStateFusionTrans(nn.Module):
    """
    • 图像编码: ResNet-18 (冻结) → 512 → 256
    • 低维 state: 2 → 256
    • 将 [img_tok, state_tok] (seq_len=2, d_model=256) 送 4 层 TransformerEncoder
    • 平均池化 → 2-维动作
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        # -- 冻结 ResNet-18 backbone --
        res = models.resnet18(weights="IMAGENET1K_V1")

        self.img_encoder = nn.Sequential(*list(res.children())[:-1])  # (B,512,1,1)
        self.img_proj = nn.Linear(512, d_model)

        # -- state 投影到同维度 --
        self.state_proj = nn.Linear(2, d_model)

        # -- Transformer 编码 seq_len = 2 --
        layer = nn.TransformerEncoderLayer(d_model=d_model,
                                           nhead=nhead,
                                           dim_feedforward=4*d_model,
                                           activation="gelu",
                                           batch_first=True)
        self.trans_encoder = nn.TransformerEncoder(layer,
                                                   num_layers=num_layers)

        # -- 输出头 --
        self.head = nn.Linear(d_model, 2)

    def forward(self, img, state):
        """
        img  : (B,3,96,96)   normalized tensor
        state: (B,2)
        """
        # img → token0
        feat_img = self.img_encoder(img).flatten(1)      # (B,512)
        tok_img  = self.img_proj(feat_img)               # (B,256)

        # state → token1
        tok_state = self.state_proj(state)               # (B,256)

        # seq = [img_tok, state_tok]
        seq = torch.stack([tok_img, tok_state], dim=1)   # (B,2,256)

        # Transformer Encoder
        seq_enc = self.trans_encoder(seq)                # (B,2,256)

        # 池化（mean）→ 动作
        fused = seq_enc.mean(dim=1)                      # (B,256)
        return self.head(fused)                          # (B,2)
