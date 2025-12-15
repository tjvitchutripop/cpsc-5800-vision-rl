from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# NOTE: This custom environment still is unable to randomize color per episode due to ManiSkill limitations.

class VisualVariationPickCube(PickCubeEnv):
    def __init__(self, cube_color=None, **kwargs):
        self.cube_color = cube_color if cube_color is not None else [1.0, 0.0, 0.0, 1.0]
        super().__init__(**kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        return super()._initialize_episode(env_idx, options)

    def _load_scene(self, options: dict):
        print("Loading scene with cube color:", self.cube_color)
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.cube_color,
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)