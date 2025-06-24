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


# モデルの定義 (動的にLinear層サイズを計算する)
class CNNModel(nn.Module):
    def __init__(self, num_class):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(5, 3, kernel_size=2, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # ダミーデータでサイズを計算する
        with torch.no_grad():
            # Conv2dの期待する入力形状 input: (N, C_in, H, W)
            # N: バッチサイズ, C_in: 入力チャンネル数, H: 高さ, W: 幅
            dummy_input = torch.zeros(1, 1, 8, 8)
            dummy_output = self._forward_features(dummy_input)
            # dummy_output.shape (1, 192) 等
            # ここで最初の数値はバッチサイズ
            # 2つ目の数値は特徴量数でこれがほしい
            self.flatten_size = dummy_output.shape[1]

        self.fc = nn.Linear(self.flatten_size, num_class)

    def _forward_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.fc(x)
        return x


model = CNNModel(num_class)

# 損失関数とオプティマイザーの設定
criterion = nn.CrossEntropyLoss()  # softmaxは含まれている
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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
# TP(真陽性) = 正しくPと予測した数 (PをPと予測した数)
# TN(真陰性) = 正しくNと予測した数 (NをNと予測した数)
# FP(偽陽性) = NをPと予測した数
# FN(偽陰性) = PをNと予測した数
#
# Precision(精度) = TP / (TP + FP) --> Pと予測した中で実際に正しかった割合
# --> 偽陽性のコストが高い場合に用いる。
# Recall (再現率) = TP / (TP + FN) --> 実際にPだった中で予測もPだった割合
# --> 偽陰性のコストが高い場合に用いる。
# F1 Score = 2 * (Precision * Recall) / (Precision + Recall) --> 精度と再現率の調和平均
# --> PrecisionとRecallのバランスが重要な場合に用いる
# Support = 各クラスのテストデータ中の実際のサンプル数
# Accuracy = 全ての正解予測数 / 全サンプル数
# --> 全体的な正解率を知りたい場合
