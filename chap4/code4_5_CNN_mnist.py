#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = load_digits()
image_shape = (1, 8, 8)  # PyTorchは(C, H, W)の順で画像を扱う
num_class = 10

y = dataset.target
X = dataset.data
X = np.array([data.reshape(image_shape) for data in X])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# データをPyTorchのテンソルに変換
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
# CrossEntropyLossはラベルを整数で扱うため、y_trainとy_testも整数型に変換
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

# モデルの定義
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=5, out_channels=3, kernel_size=2, stride=1, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    # softmaxは損失関数に含まれる
    nn.Linear(in_features=243, out_features=num_class),
)

# 損失関数とオプティマイザーの設定
criterion = nn.CrossEntropyLoss()  # softmaxは含まれている
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 訓練ループ
model.train()
for epoch in range(50):
    optimizer.zero_grad()

    # 順伝播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 逆伝播と最適化
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/8], Loss: {loss.item():.4f}")

# 予測
model.eval()
with torch.no_grad():
    predicts = model(X_test)
    predicts = torch.argmax(predicts, dim=1)  # 最大値のインデックスを取得

# numpy配列に変換して分類レポートを出力
predicts_np = predicts.cpu().numpy()
y_test_np = y_test.cpu().numpy()
print(classification_report(y_test_np, predicts_np, digits=4))
