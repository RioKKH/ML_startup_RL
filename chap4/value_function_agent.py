#!/usr/bin/env python

import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from fn_framework import FNAgent, Trainer, Observer


class ValueFunctionNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(10, 10)):
        super(ValueFunctionNet, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ValueFunctionAgent(FNAgent):
    def __init__(self, epsilon, actions, lr=0.001, device="cpu"):
        super().__init__(epsilon, actions)
        self.lr = lr
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.scaler_mean = None
        self.scaler_std = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        print(f"Using device: {self.device}")

    def save(self, model_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler_mean": self.scaler_mean.cpu()
                if self.scaler_mean is not None
                else None,
                "scaler_std": self.scaler_std.cpu()
                if self.scaler_std is not None
                else None,
                "input_size": self.model.network[0].in_features,
                "output_size": len(self.actions),
                "lr": self.lr,
                "device": str(self.device),
                "epsilon": self.epsilon,  # epsilonも保存
            },
            model_path,
        )

    @classmethod
    def load(cls, env, model_path, epsilon=None, device="cpu"):
        """
        学習済みモデルをロード
        epsilon=Noneの場合は保存されたepsilonを使用
        """
        actions = list(range(env.action_space.n))

        checkpoint = torch.load(model_path, map_location="cpu")
        device = checkpoint.get("device", "cpu")

        # epsilonが指定されていない場合は保存されたものを使用
        if epsilon is None:
            epsilon = checkpoint.get("epsilon", 0.0001)

        agent = cls(epsilon, actions, lr=checkpoint["lr"], device=device)

        # モデルを再構築
        input_size = checkpoint["input_size"]
        output_size = checkpoint["output_size"]
        agent.model = ValueFunctionNet(input_size, output_size).to(agent.device)
        agent.model.load_state_dict(checkpoint["model_state_dict"])

        # スケーラーのパラメータを復元
        if checkpoint["scaler_mean"] is not None:
            agent.scaler_mean = checkpoint["scaler_mean"].to(agent.device)
            agent.scaler_std = checkpoint["scaler_std"].to(agent.device)

        # オプティマイザーを設定
        agent.optimizer = optim.Adam(agent.model.parameters(), lr=agent.lr)

        agent.initialized = True
        print(f"Loaded model from {model_path} with epsilon={epsilon}")
        return agent

    def initialize(self, experiences):
        # 状態の次元を取得
        state_sample = experiences[0].s
        input_size = state_sample.shape[1]
        output_size = len(self.actions)

        # モデルを初期化
        self.model = ValueFunctionNet(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # StandardScalerの代替実装
        states = np.vstack([e.s for e in experiences])
        self.scaler_mean = torch.tensor(
            np.mean(states, axis=0), dtype=torch.float32
        ).to(self.device)
        self.scaler_std = torch.tensor(
            np.std(states, axis=0) + 1e-8, dtype=torch.float32
        ).to(self.device)

        # 初期学習を実行
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def _normalize(self, states):
        """StandardScalerの代替実装"""
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
        return (states - self.scaler_mean) / self.scaler_std

    def estimate(self, s):
        self.model.eval()
        with torch.no_grad():
            s_tensor = torch.tensor(s, dtype=torch.float32).to(self.device)
            s_normalized = self._normalize(s_tensor)
            estimated = self.model(s_normalized)
            return estimated.cpu().numpy()[0]

    def _predict(self, states):
        if self.initialized:
            self.model.eval()
            with torch.no_grad():
                states_tensor = torch.tensor(states, dtype=torch.float32).to(
                    self.device
                )
                states_normalized = self._normalize(states_tensor)
                predicteds = self.model(states_normalized)
                return predicteds.cpu().numpy()
        else:
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
            return predicteds

    def update(self, experiences, gamma):
        states = np.vstack([e.s for e in experiences])
        n_states = np.vstack([e.n_s for e in experiences])

        # 現在の状態と次の状態の価値を予測
        current_q_values = self._predict(states)
        future_q_values = self._predict(n_states)

        # ターゲット値を計算
        targets = current_q_values.copy()
        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future_q_values[i])
            targets[i][e.a] = reward

        # PyTorchでの学習
        self.model.train()
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)

        states_normalized = self._normalize(states_tensor)

        self.optimizer.zero_grad()
        outputs = self.model(states_normalized)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()


class CartPoleObserver(Observer):
    def reset(self):
        state, info = self._env.reset()
        return self.transform(state)

    def step(self, action):
        n_state, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        return np.array(state).reshape((1, -1))


class ValueFunctionTrainer(Trainer):
    def train(self, env, agent, episode_count=30, initial_count=-1, render=False):
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def begin_train(self, episode, agent):
        # すでに初期化済みの場合はスキップ（継続学習時）
        if not agent.initialized:
            agent.initialize(self.experiences)

    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval :]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play, load_model=None, save_as=None, continue_epsilon=None):
    if play:
        # プレイ時は表示モードで環境を作成
        env = CartPoleObserver(gym.make("CartPole-v1", render_mode="human"))
        trainer = ValueFunctionTrainer()

        # ロードするモデルを指定
        model_path = (
            load_model
            if load_model
            else trainer.logger.path_of("value_function_agent.pth")
        )

        agent = ValueFunctionAgent.load(env, model_path)
        agent.play(env)
    else:
        # 訓練時は非表示モード
        env = CartPoleObserver(gym.make("CartPole-v1"))
        trainer = ValueFunctionTrainer()

        # 保存先パスを決定
        save_path = (
            save_as if save_as else trainer.logger.path_of("value_function_agent.pth")
        )

        # 既存モデルをロードするかどうか
        if load_model:
            print(f"Loading existing model from: {load_model}")
            # 継続学習用のepsilonを指定（探索を減らすため）
            epsilon = continue_epsilon if continue_epsilon is not None else 0.05
            agent = ValueFunctionAgent.load(env, load_model, epsilon=epsilon)
            print(f"Continue training with epsilon={epsilon}")
        else:
            # 新規学習
            print("Starting new training...")
            actions = list(range(env.action_space.n))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = ValueFunctionAgent(epsilon=0.1, actions=actions, device=device)

        trained = trainer.train(env, agent)
        trainer.logger.plot("Rewards", trainer.reward_log, trainer.report_interval)
        trained.save(save_path)
        print(f"Model saved to: {save_path}")
        trainer.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")
    parser.add_argument("--load-model", type=str, help="path to load existing model")
    parser.add_argument("--save-as", type=str, help="path to save trained model")
    parser.add_argument(
        "--continue-epsilon",
        type=float,
        default=0.05,
        help="epsilon for continue training (default: 0.05)",
    )

    args = parser.parse_args()
    main(args.play, args.load_model, args.save_as, args.continue_epsilon)
