import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_env import MarketEnv


def load_data(path):
    return pd.read_csv(path, index_col=0)


def make_env(data_df, config):
    def _init():
        return MarketEnv(data_df, config)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_df = load_data(config.get("eval_data_path", config.get("data_path", "")))
    env = DummyVecEnv([make_env(data_df, config)])
    env = VecNormalize.load(config.get("output_env_norm", "./env_norm.pkl"), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.model, env=env)

    obs = env.reset()
    done = False
    rewards, weights_hist, rets_hist = [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(float(reward))
        weights_hist.append(info[0]["weights"])
        rets_hist.append(info[0]["portfolio_return"])

    equity = np.cumprod(1 + np.array(rets_hist)) - 1
    sharpe = np.mean(rets_hist) / (np.std(rets_hist) + 1e-8) * np.sqrt(52)

    print(f"Final cumulative return: {equity[-1]:.3f}")
    print(f"Annualized Sharpe: {sharpe:.2f}")

    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Week")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.get("equity_curve_path", "equity_curve.png"))
    plt.show()


if __name__ == "__main__":
    main()
