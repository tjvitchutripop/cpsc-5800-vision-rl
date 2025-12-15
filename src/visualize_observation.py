import mani_skill
import gymnasium as gym
from utils import FlattenRGBDSegObservationWrapper

env = gym.make("PushCube-v1", obs_mode="rgb+depth+segmentation")

obs = env.reset()

# Visualize rgb depth and segmentation
import matplotlib.pyplot as plt
rgb = obs[0]["sensor_data"]["base_camera"]["rgb"]
depth = obs[0]["sensor_data"]["base_camera"]["depth"]
# normalize depth
depth = (depth - depth.min()) / (depth.max() - depth.min())
segmentation = obs[0]["sensor_data"]["base_camera"]["segmentation"]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(rgb.squeeze().cpu().numpy())
axs[0].set_title("RGB")
axs[1].imshow(depth.squeeze().cpu().numpy(), cmap='gray')
axs[1].set_title("Depth")
axs[2].imshow(segmentation.squeeze().cpu().numpy(), cmap='jet')
axs[2].set_title("Segmentation")
plt.savefig("observation_visualization.png")