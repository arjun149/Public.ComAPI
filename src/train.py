import argparse
import yaml
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_env import MarketEnv


def load_data(path):
    if path is None or path == "":
        return None
    return pd.read_csv(path, index_col=0)


def make_env(data_df, config):
    def _init():
        return MarketEnv(data_df, config)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_df = load_data(config.get("data_path", ""))

    env = DummyVecEnv([make_env(data_df, config)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.get("learning_rate", 3e-4),
        gamma=config.get("gamma", 0.99),
        ent_coef=config.get("entropy_coef", 0.01),
        batch_size=config.get("batch_size", 64),
        tensorboard_log=config.get("tensorboard_log", "./logs/"),
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(config.get("output_model_path", "./ppo_market"))
    env.save_running_average(config.get("output_env_norm", "./env_norm.pkl"))
    print("✅ Training complete. Model saved.")


if __name__ == "__main__":
    main()
