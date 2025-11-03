import numpy as np
import pandas as pd
from src.gym_env import MarketEnv


def make_dummy_env():
    N, T = 3, 12
    df = pd.DataFrame(np.random.randn(T, N) * 0.01, columns=[f"ret_{i}" for i in range(N)])
    config = {
        "universe_size": N,
        "features_dim": N,
        "max_gross_leverage": 2.0,
        "per_asset_max": 1.0,
        "transaction_cost": 0.001,
        "slippage_coeff": 0.001,
    }
    env = MarketEnv(df, config)
    return env


def test_reset_and_step():
    env = make_dummy_env()
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]

    action = np.random.uniform(-1, 1, env.N)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert "weights" in info
    assert obs.shape == env.observation_space.shape
    assert np.all(np.abs(info["weights"]) <= env.config["per_asset_max"])
