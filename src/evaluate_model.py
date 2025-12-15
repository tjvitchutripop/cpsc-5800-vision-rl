#!/usr/bin/env python3
"""
Simple evaluation script to test trained PPO models.
Loads a model from /models and runs it for 10 episodes on a specified task.
"""

import argparse
import os
import torch
import gymnasium as gym
import numpy as np
from typing import Dict, Any

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from utils import FlattenRGBDSegObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from feature_extractor import NatureCNN, Theia, ResNet50, DenseNet121, EfficientNetB0
from train_ppo import Agent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (e.g., models/pickCube/pickCube_rgb.pt)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="PickCube-v1",
        choices=["PickCube-v1", "PushCube-v1", "PushT-v1"],
        help="Task environment to evaluate on"
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Include RGB modality"
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Include depth modality"
    )
    parser.add_argument(
        "--segmentation",
        action="store_true",
        help="Include segmentation modality"
    )
    parser.add_argument(
        "--state",
        action="store_true",
        default=True,
        help="Include state information (default: True)"
    )
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="nature_cnn",
        choices=["nature_cnn", "theia", "resnet50", "densenet121", "efficientnet_b0"],
        help="Feature extractor architecture"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save evaluation videos"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="eval_videos",
        help="Directory to save videos"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    return parser.parse_args()


def evaluate_model(args):
    """Main evaluation function"""
    
    print("=" * 60)
    print(f"Evaluating model: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Modalities: RGB={args.rgb}, Depth={args.depth}, Seg={args.segmentation}, State={args.state}")
    print(f"Feature Extractor: {args.feature_extractor}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    
    # Create environment
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        render_mode="all" if args.render else "rgb_array",
        sim_backend="physx_cuda"
    )
    
    env = gym.make(args.task, num_envs=1, **env_kwargs)
    
    # Wrap environment with observation wrapper
    env = FlattenRGBDSegObservationWrapper(
        env,
        rgb=args.rgb,
        depth=args.depth,
        segmentation=args.segmentation,
        state=args.state
    )
    
    # Flatten action space if needed
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    
    # Add video recording if requested
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=args.video_dir,
            save_trajectory=False,
            max_steps_per_video=args.max_steps,
            video_fps=30
        )
    
    # Get initial observation to initialize agent
    obs, _ = env.reset(seed=args.seed)
    
    # Create agent and load model
    agent = Agent(env, sample_obs=obs, feature_extractor=args.feature_extractor).to(device)
    
    # Load checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    agent.load_state_dict(checkpoint)
    agent.eval()
    print("Model loaded successfully!")
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print("\nStarting evaluation...")
    print("-" * 60)
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        for step in range(args.max_steps):
            # Get action from agent
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=args.deterministic)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward.item() if torch.is_tensor(reward) else reward
            episode_length += 1
            
            # Check if episode is done
            if terminated or truncated:
                done = True
                # Check for success (if info contains success metric)
                if "success" in info:
                    if info["success"]:
                        success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{args.num_episodes}: "
              f"Reward = {episode_reward:.3f}, Length = {episode_length}")
    
    # Print summary statistics
    print("-" * 60)
    print("\nEvaluation Summary:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min Reward: {np.min(episode_rewards):.3f}")
    print(f"  Max Reward: {np.max(episode_rewards):.3f}")
    if success_count > 0:
        print(f"  Success Rate: {success_count}/{args.num_episodes} ({100*success_count/args.num_episodes:.1f}%)")
    print("=" * 60)
    
    env.close()
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_count / args.num_episodes if success_count > 0 else 0.0,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_model(args)
