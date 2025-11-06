import gymnasium as gym
import numpy as np
import os
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from tqdm import tqdm
import argparse

ENV_NAME = 'CarRacing-v3'
RESIZE_SIZE = 64

transform = T.Compose([
    T.ToImage(),
    T.Resize((RESIZE_SIZE, RESIZE_SIZE), antialias=True),
    T.ToDtype(torch.uint8, scale=False)
])


def process_frame(frame):
    processed_tensor = transform(frame)
    return processed_tensor.permute(1, 2, 0).numpy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect data from CarRacing-v3")

    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1000,
        help='Number of full rollouts to collect. The paper uses 10,000.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset',
        help='Directory to save the rollout .npz files.'
    )
    parser.add_argument(
        '--steer_min',
        type=int,
        default=20,
        help='Minimum steps to hold a steering action.'
    )
    parser.add_argument(
        '--steer_max',
        type=int,
        default=100,
        help='Maximum steps to hold a steering action.'
    )

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Starting data collection for {args.num_rollouts} rollouts ({ENV_NAME})...")
    print(f"Data will be saved in '{args.data_dir}'")

    env = gym.make(
        ENV_NAME,
        render_mode='rgb_array',
        continuous=True
    )

    for i in tqdm(range(args.num_rollouts)):

        obs, info = env.reset()

        observations = []
        actions = []
        rewards = []
        dones = []

        done = False

        current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        current_steer_duration = 0

        while not done:

            if current_steer_duration <= 0:
                steer_choice = np.random.choice([-1.0, 0.0, 1.0])
                current_action[0] = steer_choice
                current_action[1] = 1.0
                current_action[2] = 0.0

                current_steer_duration = np.random.randint(args.steer_min, args.steer_max)

            current_steer_duration -= 1

            next_obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated

            observations.append(process_frame(obs))
            actions.append(current_action)
            rewards.append(reward)
            dones.append(done)

            obs = next_obs

        filename = os.path.join(args.data_dir, f'rollout_{i}.npz')

        np.savez_compressed(
            filename,
            observations=np.array(observations, dtype=np.uint8),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=bool)
        )

    env.close()
    print(f"\nData collection complete. {args.num_rollouts} rollouts saved in '{args.data_dir}'.")