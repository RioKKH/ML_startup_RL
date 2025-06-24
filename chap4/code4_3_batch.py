#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn

# 2層ニューラルテネットワーク
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=4),
    nn.Sigmoid(),
    nn.Linear(in_features=4, out_features=4),
)

# バッチサイズ = 3のデータを作成 (xの次元は2)
batch = torch.from_numpy(np.random.rand(3, 2)).float()

# 推論の実行
model.eval()  # 評価モードに設定
with torch.no_grad():  # 勾配計算を無効化
    y = model(batch)

print(y.shape)  # (3, 4)になる
