"""Picklable training entry points for the cross-validation protocol:
``fn(train_weather, cfg: dict, seed) -> (policy, learning_curve)``."""
from .ppo import PPOConfig, train_ppo
from .tabular_q import train_tabular_q


def tabular(weather, cfg, seed):
    return train_tabular_q(weather, seed=seed, **cfg)


def ppo(weather, cfg, seed):
    return train_ppo(weather, PPOConfig(**cfg, seed=seed))
