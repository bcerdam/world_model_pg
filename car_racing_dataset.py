import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import time

ENV_NAME = 'CarRacing-v3'

stop_loop = False

env = gym.make(ENV_NAME, render_mode='rgb_array', continuous=True)
obs, info = env.reset()

steer_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0.0, description='Steer:')
gas_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.1, value=0.0, description='Gas:')
brake_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.1, value=0.0, description='Brake:')
stop_button = widgets.Button(description='Stop Simulation')


def on_stop_button_clicked(b):
    global stop_loop
    stop_loop = True


stop_button.on_click(on_stop_button_clicked)

print("--- Controls ---")
display(steer_slider, gas_slider, brake_slider, stop_button)
print("\n--- Game View ---")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

try:
    while not stop_loop:
        s = steer_slider.value
        g = gas_slider.value
        b = brake_slider.value

        action = np.array([s, g, b], dtype=np.float32)

        frame, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Rollout Finished. Resetting environment.")
            obs, info = env.reset()

        clear_output(wait=True)

        print("--- Controls ---")
        display(steer_slider, gas_slider, brake_slider, stop_button)
        print("\n--- Game View ---")
        print(f"Action: [S:{s: .2f}, G:{g: .2f}, B:{b: .2f}] | Reward: {reward: .4f}")

        ax.imshow(frame)
        ax.axis('off')
        display(fig)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nLoop interrupted.")
finally:
    env.close()
    clear_output(wait=True)
    print("Simulation stopped and environment closed.")