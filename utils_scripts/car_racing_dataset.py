import gymnasium as gym
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import argparse
import multiprocessing
from functools import partial

ENV_NAME = 'CarRacing-v3'
RESIZE_SIZE = 64
WARMUP_STEPS = 50

original_height = 96
original_width = 96
crop_height = 84


def process_frame(frame):
    img = Image.fromarray(frame)
    crop_box = (0, 0, original_width, crop_height)
    img_cropped = img.crop(crop_box)
    img_resized = img_cropped.resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.BILINEAR)
    return np.array(img_resized)


def run_rollout(rollout_id, args):
    env = gym.make(
        ENV_NAME,
        render_mode='rgb_array',
        continuous=True
    )

    obs, info = env.reset()

    warmup_action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(WARMUP_STEPS):
        obs, _, _, _, _ = env.step(warmup_action)

    observations = []
    actions = []
    rewards = []
    dones = []

    done = False

    current_action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    frames_remaining = np.random.randint(args.straight_duration_min, args.straight_duration_max)
    state = 'STRAIGHT'

    while not done:

        if frames_remaining <= 0:

            if np.random.rand() < args.brake_prob and state != 'BRAKE':
                state = 'BRAKE'
                current_action[0] = 0.0
                current_action[1] = 0.0
                current_action[2] = 1.0
                frames_remaining = np.random.randint(args.brake_duration_min, args.brake_duration_max)

            elif state == 'STRAIGHT' or state == 'BRAKE':
                state = 'TURN'
                current_action[0] = np.random.uniform(-1.0, 1.0)
                current_action[1] = 1.0
                current_action[2] = 0.0
                frames_remaining = np.random.randint(args.turn_duration_min, args.turn_duration_max)

            elif state == 'TURN':
                state = 'STRAIGHT'
                current_action[0] = 0.0
                current_action[1] = 1.0
                current_action[2] = 0.0
                frames_remaining = np.random.randint(args.straight_duration_min, args.straight_duration_max)

        frames_remaining -= 1

        next_obs, reward, terminated, truncated, info = env.step(current_action)
        done = terminated or truncated

        observations.append(process_frame(obs))
        actions.append(current_action.copy())
        rewards.append(reward)
        dones.append(done)

        obs = next_obs

    filename = os.path.join(args.data_dir, f'rollout_{rollout_id}.npz')

    np.savez_compressed(
        filename,
        observations=np.array(observations, dtype=np.uint8),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool)
    )

    env.close()
    return rollout_id


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect data from CarRacing-v3")

    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='Number of full rollouts to collect.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        # --- PATH UPDATED ---
        default='data/vae_dataset',
        help='Directory to save the rollout .npz files.'
    )
    parser.add_argument(
        '--turn_duration_min',
        type=int,
        default=2,
        help='Minimum steps to hold a steering action.'
    )
    parser.add_argument(
        '--turn_duration_max',
        type=int,
        default=5,
        help='Maximum steps to hold a steering action.'
    )
    parser.add_argument(
        '--straight_duration_min',
        type=int,
        default=1,
        help='Minimum steps to drive straight.'
    )
    parser.add_argument(
        '--straight_duration_max',
        type=int,
        default=2,
        help='Maximum steps to drive straight.'
    )
    parser.add_argument(
        '--brake_prob',
        type=float,
        default=0.1,
        help='Probability of braking after a phase ends.'
    )
    parser.add_argument(
        '--brake_duration_min',
        type=int,
        default=1,
        help='Minimum steps to hold the brake.'
    )
    parser.add_argument(
        '--brake_duration_max',
        type=int,
        default=2,
        help='Maximum steps to hold the brake.'
    )

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Starting data collection for {args.num_rollouts} rollouts ({ENV_NAME})...")
    print(f"Data will be saved in '{args.data_dir}'")

    if args.num_rollouts > 1:
        num_workers = multiprocessing.cpu_count()
        print(f"Using {num_workers} CPU workers.")

        worker_fn = partial(run_rollout, args=args)
        rollout_ids = range(args.num_rollouts)

        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_fn, rollout_ids),
                total=args.num_rollouts
            ))

        print(f"\nData collection complete. {len(results)} rollouts saved in '{args.data_dir}'.")

    else:
        print("Running in single-threaded mode for 1 rollout...")
        for i in tqdm(range(args.num_rollouts)):
            run_rollout(i, args)
        print(f"\nData collection complete. {args.num_rollouts} rollouts saved in '{args.data_dir}'.")