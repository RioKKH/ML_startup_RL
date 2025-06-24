#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

# モデルの定義
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=4),
)

# 重みとバイアスの取得と形状表示
weight = model[0].weight.data
bias = model[0].bias.data

print(f"Weight shape is {weight.shape}")
print(f"Bias shape is {bias.shape}")

# 推論の実行
model.eval()  # 評価モードに設定
x = torch.from_numpy(np.random.rand(1, 2)).float()
with torch.no_grad():  # 勾配計算を無効化
    y = model(x)

print(f"x is ({x.shape}) and y is ({y.shape})")
