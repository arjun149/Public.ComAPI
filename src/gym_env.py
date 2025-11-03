import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class MarketEnv(gym.Env):
    """
    Custom Market environment for long-short weekly portfolio rebalancing.
    Works with weekly return data or synthetic fallback.

    Observation: flattened feature vector + current weights + cash
    Action: continuous vector -> target portfolio weights (long-short)
    Reward: next-week portfolio return minus transaction cost & slippage
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, data_df: pd.DataFrame | None, config: dict):
        super().__init__()
        self.config = config
        self.data_df = (
            data_df if data_df is not None else self._make_dummy_data(config)
        )
        self.N = config.get("universe_size", 5)

        # --- Gym spaces ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32
        )
        obs_dim = config.get("features_dim", 10) + self.N + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- State vars ---
        self._current_step = 0
        self._weights = np.zeros(self.N, dtype=np.float32)
        self._cash = 1.0
        self._done = False

    # ------------------------------------------------------------------
    # Core Gym methods
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._weights = np.zeros(self.N, dtype=np.float32)
        self._cash = self.config.get("start_cash", 1.0)
        self._done = False
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # map to portfolio weights
        target_w = self._map_action_to_weights(action)

        # compute turnover & costs
        turnover = np.sum(np.abs(target_w - self._weights))
        cost = self.config.get("transaction_cost", 0.001) * turnover
        slippage = self.config.get("slippage_coeff", 0.001) * turnover

        # compute portfolio return for next step
        returns = self._get_returns_for_step(self._current_step + 1)
        port_ret = np.dot(target_w, returns)

        # update cash and weights
        self._cash *= (1.0 + port_ret)
        self._cash -= (cost + slippage)
        self._weights = target_w

        # reward (risk-adjusted if you want)
        reward = (
            port_ret
            - cost
            - slippage
            - self.config.get("margin_interest", 0.0) * np.sum(np.abs(target_w))
        )

        self._current_step += 1
        terminated = self._current_step >= len(self.data_df) - 1
        truncated = False
        info = {
            "portfolio_return": float(port_ret),
            "turnover": float(turnover),
            "weights": self._weights.copy(),
        }

        return self._get_observation(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _map_action_to_weights(self, action):
        raw = np.tanh(action)
        max_lev = self.config.get("max_gross_leverage", 2.0)
        if np.sum(np.abs(raw)) > 0:
            raw = raw / np.sum(np.abs(raw)) * max_lev
        per_asset_max = self.config.get("per_asset_max", 1.0)
        return np.clip(raw, -per_asset_max, per_asset_max)

    def _get_returns_for_step(self, step):
        cols = self.config.get("return_cols", [f"ret_{i}" for i in range(self.N)])
        return self.data_df.iloc[step][cols].to_numpy(dtype=np.float32)

    def _extract_features(self, step):
        cols = self.config.get("feature_cols", [])
        if not cols:
            # fallback: use past returns as simple features
            cols = [f"ret_{i}" for i in range(self.N)]
        return self.data_df.iloc[step][cols].to_numpy(dtype=np.float32)

    def _get_observation(self):
        feats = self._extract_features(self._current_step)
        obs = np.concatenate([feats, self._weights, [self._cash]], axis=0)
        return obs.astype(np.float32)

    def _make_dummy_data(self, config):
        """Creates a small synthetic dataset for testing."""
        N = config.get("universe_size", 5)
        T = config.get("dummy_len", 104)
        rets = np.random.randn(T, N) * 0.01
        df = pd.DataFrame(rets, columns=[f"ret_{i}" for i in range(N)])
        return df

    def render(self):
        print(
            f"Step {self._current_step} | Cash {self._cash:.3f} | Weights {self._weights}"
        )

    def close(self):
        pass
