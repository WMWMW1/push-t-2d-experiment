# bc_model.py
import torch
import torch.nn as nn
from torchvision import models

class ResNetStateFusionTrans(nn.Module):
    """
    • 图像编码: ResNet-18 → 512 → 256
    • 低维 state: 2 → 256
    • 将 [img_tok, state_tok] (seq_len=2) 送 4 层 Transformer
    • 池化 → head → 输出 chunk_size 步动作 (B, chunk_size, 2)
    """
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size

        # -- ResNet-18 backbone (可选冻结/解冻) --
        res = models.resnet18(weights="IMAGENET1K_V1")
        # 如果你要解冻，删掉下面这两行：
        # for p in res.parameters():
        #     p.requires_grad = False

        self.img_encoder = nn.Sequential(*list(res.children())[:-1])  # (B,512,1,1)
        self.img_proj    = nn.Linear(512, d_model)

        # -- state 投影到同维 --
        self.state_proj = nn.Linear(2, d_model)

        # -- Transformer Encoder (seq_len=2) --
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            activation="gelu",
            batch_first=True
        )
        self.trans_encoder = nn.TransformerEncoder(layer,
                                                   num_layers=num_layers)

        # -- head: 输出 chunk_size * 2 维 --
        self.head = nn.Linear(d_model, 2 * chunk_size)

    def forward(self, img, state):
        """
        img   : (B,3,H,W)
        state : (B,2)
        returns: (B, chunk_size, 2)
        """
        # [img]
        feat_img = self.img_encoder(img).flatten(1)  # (B,512)
        tok_img  = self.img_proj(feat_img)           # (B,256)

        # [state]
        tok_state = self.state_proj(state)           # (B,256)

        # 合并 seq
        seq = torch.stack([tok_img, tok_state], dim=1)  # (B,2,256)
        seq_enc = self.trans_encoder(seq)               # (B,2,256)

        # 池化 → head
        fused = seq_enc.mean(dim=1)  # (B,256)
        out   = self.head(fused)     # (B, chunk_size*2)
        return out.view(-1, self.chunk_size, 2)  # (B,chunk_size,2)
