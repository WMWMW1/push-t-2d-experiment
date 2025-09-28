#!/usr/bin/env python3
import torch
from bc_model import ResNetStateFusionTrans

def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    # 实例化模型（使用默认的 d_model=256, num_layers=4）
    model = ResNetStateFusionTrans()

    total, trainable = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
