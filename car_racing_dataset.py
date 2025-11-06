import gymnasium as gym
import numpy as np
import os
# THE FIX IS HERE: We only import 'Image' from 'PIL'
from PIL import Image
from tqdm import tqdm
import argparse

# THE FIX IS HERE: We no longer import 'torch' or 'torchvision' in this script
# as they are not needed for data collection.

ENV_NAME = 'CarRacing-v3'
RESIZE_SIZE = 64
WARMUP_STEPS = 50

original_height = 96
original_width = 96
crop_height = 84


# THE FIX IS HERE: The 'transform' block is removed.

# THE FIX IS HERE: This function now uses PIL and NumPy,
# which are stable and reliable.
def process_frame(frame):
    img = Image.fromarray(frame)
    crop_box = (0, 0, original_width, crop_height)
    img_cropped = img.crop(crop_box)
    img_resized = img_cropped.resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.BILINEAR)
    return np.array(img_resized)


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

        warmup_action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(WARMUP_STEPS):
            obs, _, _, _, _ = env.step(warmup_action)

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