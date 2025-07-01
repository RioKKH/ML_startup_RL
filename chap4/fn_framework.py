#!/usr/bin/env python

import os
import io
import re
from collections import namedtuple
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt


Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])


class FNAgent:
    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False

    def save(self, model_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_class": self.model.__class__,
                "model_args": getattr(self.model, "_init_args", {}),
            },
            model_path,
        )

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)

        checkpoint = torch.load(model_path, map_location="cpu")
        agent.model = checkpoint["model_state_dict"]
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        raise NotImplementedError("You have to implement initialize method.")

    def estimate(self, s):
        raise NotImplementedError("You have to implement estimate method.")

    def update(self, experiences, gamma):
        raise NotImplementedError("You have to implement update method.")

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)
            if self.estimate_probs:
                action = np.random.choice(self.actions, size=1, p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)

    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print(f"Get reward {episode_reward}")


class Trainer:
    def __init__(
        self, buffer_size=1024, batch_size=32, gamma=0.9, report_interval=10, log_dir=""
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []

    @property
    def trainer_name(self):
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked

    def train_loop(
        self,
        env,
        agent,
        episode=200,
        initial_count=-1,
        render=False,
        observe_interval=0,
    ):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []

        for i in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            episode_reward = 0  # エピソード報酬を追跡
            self.episode_begin(i, agent)

            while not done:
                if render:
                    env.render()
                if (
                    self.training
                    and observe_interval > 0
                    and (
                        self.training_count == 1
                        or self.training_count % observe_interval == 0
                    )
                ):
                    frames.append(s)

                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward  # エピソード報酬を累積

                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)
                if not self.training and len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True

                self.step(i, step_count, agent, e)
                s = n_state
                step_count += 1
            else:
                # エピソード報酬を直接使用
                self.reward_log.append(episode_reward)
                self.episode_end(i, step_count, agent)

                if not self.training and initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True

                if self.training:
                    if len(frames) > 0:
                        self.logger.write_image(self.training_count, frames)
                        frames = []
                    self.training_count += 1

    def episode_begin(self, episode, agent):
        pass

    def begin_train(self, episode, agent):
        pass

    def step(self, episode, step_count, agent, experience):
        pass

    def episode_end(self, episode, step_count, agent):
        pass

    def is_event(self, count, interval):
        return True if count != 0 and count % interval == 0 else False

    def get_recent(self, count):
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]


class Observer:
    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        # Gymnasium API対応
        result = self._env.reset()
        if isinstance(result, tuple):
            state, info = result
        else:
            state = result
        return self.transform(state)

    def render(self):
        self._env.render()

    def step(self, action):
        # Gymnasium API対応
        result = self._env.step(action)
        if len(result) == 5:  # 新しいGymnasium
            n_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # 古いGym
            n_state, reward, done, info = result
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        raise NotImplementedError("You have to implement transform method.")


class Logger:
    def __init__(self, log_dir="", dir_name=""):
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if dir_name:
            self.log_dir = os.path.join(self.log_dir, dir_name)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)

    def set_model(self, model):
        pass

    def path_of(self, file_name):
        return os.path.join(self.log_dir, file_name)

    def describe(self, name, values, episode=-1, step=-1):
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = f"{name} is {mean} (+/-{std})"
        if episode > 0:
            print(f"At episode {episode}, {desc}")
        elif step > 0:
            print(f"At step {step}, {desc}")

    def plot(self, name, values, interval=10):
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            _values = values[i : (i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title(f"{name} History")
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
        plt.plot(
            indices,
            means,
            "o-",
            color="g",
            label=f"{name.lower()} per {interval} episode",
        )
        plt.legend(loc="best")
        plt.show()

    def write(self, index, name, value):
        self.writer.add_scalar(name, value, index)
        self.writer.flush()

    def write_image(self, index, frames):
        last_frames = [f[:, :, -1] for f in frames]
        if np.min(last_frames[-1]) < 0:
            scale = 127 / np.abs(last_frames[-1]).max()
            offset = 128
        else:
            scale = 255 / np.max(last_frames[-1])
            offset = 0

        tag = f"frames_at_training_{index}"

        for i, f in enumerate(last_frames):
            normalized_frame = (f * scale + offset) / 255.0
            frame_tensor = torch.from_numpy(normalized_frame).unsqueeze(0).float()
            self.writer.add_image(f"{tag}/frame_{i}", frame_tensor, index)

        self.writer.flush()

    def close(self):
        """TensorBoard writerを閉じる"""
        self.writer.close()
