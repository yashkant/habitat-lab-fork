from typing import Any, Dict

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.nav import NavMeshSettings
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum
from habitat.config.default import Config
import os
from habitat_sim.physics import MotionType
import numpy as np
from habitat.config.default import _C, CN
from habitat.core.registry import registry
from actions import raycast
from gym import spaces

import habitat_sim
from habitat.config.default import CN, Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import PointGoalSensor

@registry.register_simulator(name="CosRearrangementSim-v0")
class CosRearrangementSim(HabitatSim):
    r"""Simulator wrapper over habitat-sim with
    object rearrangement functionalities.
    """

    def __init__(self, config: Config) -> None:
        self.did_reset = False
        super().__init__(config=config)
        self.grip_offset = np.eye(4)

        agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_config = self._get_agent_config(agent_id)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.RADIUS
        self.navmesh_settings.agent_height = agent_config.HEIGHT

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)
        self._initialize_objects()

    def reset(self):
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)

    def _initialize_objects(self):
        objects = self.habitat_config.objects[0]
        obj_attr_mgr = self.get_object_template_manager()
        obj_attr_mgr.load_configs(
            str(os.path.join(data_path, "test_assets/objects"))
        )
        # first remove all existing objects
        existing_object_ids = self.get_existing_object_ids()

        if len(existing_object_ids) > 0:
            for obj_id in existing_object_ids:
                self.remove_object(obj_id)

        self.sim_object_to_objid_mapping = {}
        self.objid_to_sim_object_mapping = {}

        if objects is not None:
            object_template = objects["object_template"]
            object_pos = objects["position"]
            object_rot = objects["rotation"]

            object_template_id = obj_attr_mgr.load_object_configs(
                object_template
            )[0]
            object_attr = obj_attr_mgr.get_template_by_ID(object_template_id)
            obj_attr_mgr.register_template(object_attr)

            object_id = self.add_object_by_handle(object_attr.handle)
            self.sim_object_to_objid_mapping[object_id] = objects["object_id"]
            self.objid_to_sim_object_mapping[objects["object_id"]] = object_id

            self.set_translation(object_pos, object_id)
            if isinstance(object_rot, list):
                object_rot = quat_from_coeffs(object_rot)

            object_rot = quat_to_magnum(object_rot)
            self.set_rotation(object_rot, object_id)

            self.set_object_motion_type(MotionType.STATIC, object_id)

        # Recompute the navmesh after placing all the objects.
        self.recompute_navmesh(self.pathfinder, self.navmesh_settings, True)

    def _sync_gripped_object(self, gripped_object_id):
        r"""
        Sync the gripped object with the object associated with the agent.
        """
        if gripped_object_id != -1:
            agent_body_transformation = (
                self._default_agent.scene_node.transformation
            )
            self.set_transformation(
                agent_body_transformation, gripped_object_id
            )
            translation = agent_body_transformation.transform_point(
                np.array([0, 2.0, 0])
            )
            self.set_translation(translation, gripped_object_id)

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)

    def step(self, action: int):
        dt = 1 / 60.0
        self._num_total_frames += 1
        collided = False
        gripped_object_id = self.gripped_object_id

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]

        if action_spec.name == "grab_or_release_object_under_crosshair":
            # If already holding an agent
            if gripped_object_id != -1:
                agent_body_transformation = (
                    self._default_agent.scene_node.transformation
                )
                T = np.dot(agent_body_transformation, self.grip_offset)

                self.set_transformation(T, gripped_object_id)

                position = self.get_translation(gripped_object_id)

                if self.pathfinder.is_navigable(position):
                    self.set_object_motion_type(
                        MotionType.STATIC, gripped_object_id
                    )
                    gripped_object_id = -1
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )
            # if not holding an object, then try to grab
            else:
                gripped_object_id = raycast(
                    self,
                    action_spec.actuation.visual_sensor_name,
                    crosshair_pos=action_spec.actuation.crosshair_pos,
                    max_distance=action_spec.actuation.amount,
                )

                # found a grabbable object.
                if gripped_object_id != -1:
                    agent_body_transformation = (
                        self._default_agent.scene_node.transformation
                    )

                    self.grip_offset = np.dot(
                        np.array(agent_body_transformation.inverted()),
                        np.array(self.get_transformation(gripped_object_id)),
                    )
                    self.set_object_motion_type(
                        MotionType.KINEMATIC, gripped_object_id
                    )
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )

        else:
            collided = self._default_agent.act(action)
            self._last_state = self._default_agent.get_state()

        # step physics by dt
        super().step_world(dt)

        # Sync the gripped object after the agent moves.
        self._sync_gripped_object(gripped_object_id)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations()
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations


@registry.register_sensor
class GrippedObjectSensor(Sensor):
    cls_uuid = "gripped_object_id"

    def __init__(
        self, *args: Any, sim: CosRearrangementSim, config: Config, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):

        return spaces.Discrete(len(self._sim.get_existing_object_ids()))

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: Episode,
        *args: Any,
        **kwargs: Any,
    ):
        obj_id = self._sim.sim_object_to_objid_mapping.get(
            self._sim.gripped_object_id, -1
        )
        return obj_id


@registry.register_sensor
class ObjectPosition(PointGoalSensor):
    cls_uuid: str = "object_position"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        object_id = self._sim.get_existing_object_ids()[0]
        object_position = self._sim.get_translation(object_id)
        pointgoal = self._compute_pointgoal(
            agent_position, rotation_world_agent, object_position
        )
        return pointgoal


@registry.register_sensor
class ObjectGoal(PointGoalSensor):
    cls_uuid: str = "object_goal"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        goal_position = np.array(episode.goals.position, dtype=np.float32)

        point_goal = self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
        return point_goal


if __name__ == "__main__":
    _C.TASK.ACTIONS.GRAB_RELEASE = CN()
    _C.TASK.ACTIONS.GRAB_RELEASE.TYPE = "GrabOrReleaseAction"
    _C.SIMULATOR.CROSSHAIR_POS = [128, 160]
    _C.SIMULATOR.GRAB_DISTANCE = 2.0
    _C.SIMULATOR.VISUAL_SENSOR = "rgb"
