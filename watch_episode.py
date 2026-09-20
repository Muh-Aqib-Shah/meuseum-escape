import pickle, time
import matplotlib.pyplot as plt
from mansion_heist_env import MansionHeistEnv

with open("models/pi_vi_results.pkl", "rb") as f:
    results = pickle.load(f)
with open("models/dynamics_model.pkl", "rb") as f:
    model = pickle.load(f)

env = MansionHeistEnv(size=model["size"], slip_prob=0.2,
                       loot_pos=model["loot_pos"], exit_pos=model["exit_pos"])
policy = results["pi_policy"]

obs, _ = env.reset()
plt.ion()  # interactive mode, so the window updates in place
terminated = truncated = False
while not (terminated or truncated):
    fig = env.render()
    plt.pause(0.4)          # short pause so you can see each step
    plt.close(fig)
    action = policy[obs]
    obs, reward, terminated, truncated, _ = env.step(action)

print("Episode ended. Terminated:", terminated, "Reward:", reward)