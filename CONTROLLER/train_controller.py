import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cma
import gymnasium as gym
import argparse
import os
import multiprocessing
from functools import partial
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM
from RNN.rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM
from controller import Controller

ENV_NAME = 'CarRacing-v3'
RESIZE_SIZE = 64
WARMUP_STEPS = 50


def process_frame(frame):
    img = Image.fromarray(frame)
    crop_box = (0, 0, 96, 84)
    img_cropped = img.crop(crop_box)
    img_resized = img_cropped.resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.BILINEAR)
    return np.array(img_resized)


def rollout(params, args):
    torch.set_grad_enabled(False)

    vae = VAE(z_dim=LATENT_DIM)
    vae.load_state_dict(torch.load(args.vae_path, map_location='cpu', weights_only=True))
    vae.eval()

    rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM)
    rnn.load_state_dict(torch.load(args.rnn_path, map_location='cpu', weights_only=True))
    rnn.eval()

    controller = Controller(LATENT_DIM, HIDDEN_DIM, ACTION_DIM)
    controller.set_flat_params(params)
    controller.eval()

    transform = T.Compose([T.ToPILImage(), T.ToTensor()])

    env = gym.make(ENV_NAME, render_mode='rgb_array', continuous=True)
    obs, _ = env.reset()

    total_reward = 0
    hidden_state = None

    warmup_action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(WARMUP_STEPS):
        obs, _, _, _, _ = env.step(warmup_action)

    obs_processed = process_frame(obs)
    obs_tensor = transform(obs_processed).unsqueeze(0)

    mu, logvar = vae.encode(obs_tensor)
    z = vae.reparameterize(mu, logvar)

    done = False
    while not done:
        if hidden_state is None:
            hidden_state = (torch.zeros(1, 1, HIDDEN_DIM), torch.zeros(1, 1, HIDDEN_DIM))

        action_tensor = controller(z, hidden_state[0].squeeze(0))
        action = action_tensor.numpy().flatten()

        # Action mapping: Steer [-1, 1], Gas [0, 1], Brake [0, 1]
        # Tanh gives [-1, 1]. We keep steer. We map gas/brake to [0, 1]
        action[1] = (action[1] + 1) / 2
        action[2] = (action[2] + 1) / 2

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Penalize grass driving heavily to encourage track following
        # CarRacing gives -0.1 every frame. Grass gives usually less.
        if reward < 0:
            # Give a bit of help to survive
            reward += 0.05

        total_reward += reward

        obs_processed = process_frame(obs)
        obs_tensor = transform(obs_processed).unsqueeze(0)

        mu, logvar = vae.encode(obs_tensor)
        z = vae.reparameterize(mu, logvar)

        action_rnn = action_tensor.view(1, 1, ACTION_DIM)
        z_rnn = z.unsqueeze(1)

        _, hidden_state = rnn(z_rnn, action_rnn, hidden_state)

    return -total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_path', type=str, default='model_checkpoints/vae_checkpoints/vae_epoch_10.pth')
    parser.add_argument('--rnn_path', type=str, default='model_checkpoints/rnn_checkpoints/rnn_epoch_4.pth')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints/controller_checkpoints/controller_output')
    parser.add_argument('--pop_size', type=int, default=16)
    parser.add_argument('--n_generations', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    temp_controller = Controller(LATENT_DIM, HIDDEN_DIM, ACTION_DIM)
    initial_params = temp_controller.get_flat_params()

    es = cma.CMAEvolutionStrategy(initial_params, 0.1, {'popsize': args.pop_size})

    print(f"Starting CMA-ES training for {args.n_generations} generations...")

    pool = multiprocessing.Pool(args.num_workers)

    try:
        for gen in range(args.n_generations):
            solutions = es.ask()

            rewards = pool.map(partial(rollout, args=args), solutions)

            es.tell(solutions, rewards)
            es.logger.add()
            es.disp()

            if (gen + 1) % 5 == 0:
                best_params = es.result.xbest
                save_path = os.path.join(args.output_dir, f'controller_gen_{gen + 1}.pth')

                save_model = Controller(LATENT_DIM, HIDDEN_DIM, ACTION_DIM)
                save_model.set_flat_params(best_params)
                torch.save(save_model.state_dict(), save_path)
                print(f"Saved checkpoint: {save_path}")

    finally:
        pool.close()
        pool.join()

    best_params = es.result.xbest
    save_path = os.path.join(args.output_dir, 'controller_final.pth')
    final_model = Controller(LATENT_DIM, HIDDEN_DIM, ACTION_DIM)
    final_model.set_flat_params(best_params)
    torch.save(final_model.state_dict(), save_path)
    print("Training complete.")