#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split

# from sklearn.datasets import fetch_california_housing
# from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
y = target
X = data

# dataset = load_boston()
# y = dataset.target
# X = dataset.data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# データをPyTorchテンソルに変換する
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float().view(-1, 1)
y_test = torch.from_numpy(y_test).float().view(-1, 1)

# モデルの定義
model = nn.Sequential(
    nn.BatchNorm1d(13),
    nn.Linear(in_features=13, out_features=13),
    nn.Softplus(),
    nn.Linear(in_features=13, out_features=1),
    # nn.BatchNorm1d(8),
    # nn.Linear(in_features=8, out_features=13),
)

# 損失関数とオプティマイザーの設定
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# L1正則化の為のパラメータ
l1_lambda = 0.01

# 訓練ループ
# model.fit() --> 手動で訓練ループを実装する必要がある
# --> その代わりより細かい制御が可能になる
# --> pytorch lightningなどのライブラリを使うと自動化できる
model.train()
for epoch in range(20):
    optimizer.zero_grad()

    # 順伝播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # L1正則化を追加 (最初の線形層の重みに対して)
    l1_penalty = 0
    for name, param in model.named_parameters():
        if "weight" in name and "1.weight" in name:  # 最初のLinear層の重み
            l1_penalty += torch.sum(torch.abs(param))

    loss += l1_lambda * l1_penalty

    # 逆伝播と最適化
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/8], Loss: {loss.item():.4f}")

# 予測
model.eval()
with torch.no_grad():
    predicts = model(X_test)

# 結果をnumpy配列に変換
predicts_np = predicts.cpu().numpy().reshape(-1)
y_test_np = y_test.cpu().numpy().reshape(-1)

result = pd.DataFrame({"predict": predicts_np, "actual": y_test_np})
limit = np.max(y_test_np)

result.plot.scatter(x="actual", y="predict", xlim=(0, limit), ylim=(0, limit))
plt.show()
