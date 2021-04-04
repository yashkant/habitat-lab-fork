import os
from typing import Any, Dict

import numpy as np
from gym import spaces
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum
import magnum as mn

from habitat.sims.cos_eor.actions import raycast
from habitat.config.default import CN, Config
from habitat.config.default import _C
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import PointGoalSensor
from orp.obj_loaders import load_articulated_objs, load_objs, place_viz_objs, init_art_objs
from habitat.utils.geometry_utils import (quaternion_from_coeff)


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

    def get_both_existing_object_ids(self):
        non_art_ids = super(CosRearrangementSim, self).get_existing_object_ids()
        art_ids = self.get_existing_articulated_object_ids()
        return {
            "art": art_ids,
            "non_art": non_art_ids
        }

    def get_both_existing_object_ids_with_positions(self):
        non_art_ids = super(CosRearrangementSim, self).get_existing_object_ids()
        non_art_pos = [self.get_translation(id) for id in non_art_ids]

        art_ids = self.get_existing_articulated_object_ids()
        art_pos = [np.array(self.get_articulated_object_root_state(id).translation) for id in art_ids]

        return {
            "art": art_ids,
            "art_pos": art_pos,
            "non_art": non_art_ids,
            "non_art_pos": non_art_pos,
        }

    def _initialize_objects(self):
        # There are two types of objects -- articulate and non-articulate
        _object = self.habitat_config.objects[0]
        self.sim_object_to_objid_mapping = {}
        self.objid_to_sim_object_mapping = {}

        if not _object["is_articulate"]:
            obj_attr_mgr = self.get_object_template_manager()

            obj_attr_mgr.load_configs(
                str(os.path.join("./habitat-lab/data/", "test_assets/objects"))
            )
            # first remove all existing objects
            existing_object_ids = self.get_existing_object_ids()

            if len(existing_object_ids) > 0:
                for obj_id in existing_object_ids:
                    self.remove_object(obj_id)

            if _object is not None:
                object_template = _object["object_template"]
                object_pos = _object["position"]
                object_rot = _object["rotation"]

                object_template_id = obj_attr_mgr.load_object_configs(object_template)[0]
                object_attr = obj_attr_mgr.get_template_by_ID(object_template_id)
                obj_attr_mgr.register_template(object_attr)

                object_id = self.add_object_by_handle(object_attr.handle)
                self.sim_object_to_objid_mapping[object_id] = _object["object_id"]
                self.objid_to_sim_object_mapping[_object["object_id"]] = object_id

                self.set_translation(object_pos, object_id)
                if isinstance(object_rot, list):
                    object_rot = quat_from_coeffs(object_rot)

                object_rot = quat_to_magnum(object_rot)
                self.set_rotation(object_rot, object_id)

                self.set_object_motion_type(MotionType.STATIC, object_id)

        else:
            # remove objects
            existing_object_ids = self.get_existing_articulated_object_ids()
            if len(existing_object_ids) > 0:
                for obj_id in existing_object_ids:
                    self.remove_articulated_object(obj_id)

            obj_rot, obj_translation = mn.Quaternion(_object["rotation"][:3], _object["rotation"][3]), mn.Vector3(_object["position"])
            obj_transform = getattr(mn.Matrix4, "from")(obj_rot.to_matrix(), obj_translation)
            obj_type = 2  # dynamic

            # insert start position object
            art_obj_data = [_object["object_template"], [obj_transform, obj_type]]
            object_id = load_articulated_objs([art_obj_data], self, frozen=True)[0]

            self.sim_object_to_objid_mapping[object_id] = _object["object_id"]
            self.objid_to_sim_object_mapping[_object["object_id"]] = object_id

        # Recompute the navmesh after placing all the objects.
        self.recompute_navmesh(self.pathfinder, self.navmesh_settings, True)

    def get_object_type(self, object_id):
        ids_dict = self.get_both_existing_object_ids()
        if object_id in ids_dict["art"]:
            return "art"
        else:
            return "non_art"

    def _sync_gripped_object(self, gripped_object_id):
        r"""
        Sync the gripped object with the object associated with the agent.
        """

        if gripped_object_id != -1:
            agent_transform = self._default_agent.scene_node.transformation
            obj_translation = agent_transform.transform_point(np.array([0, 2.0, 0]))
            object_transform = getattr(mn.Matrix4, 'from')\
                (agent_transform.rotation(), mn.Vector3(obj_translation))

            # holding it on the head of the agent
            ids_dict = self.get_both_existing_object_ids()

            if gripped_object_id in ids_dict["art"]:
                self.set_articulated_object_root_state(gripped_object_id, object_transform)
            else:
                self.set_transformation(
                    agent_transform, gripped_object_id
                )
                self.set_translation(obj_translation, gripped_object_id)

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)

    def debug_frame(self):
        import cv2
        obs = self.get_sensor_observations()
        rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGBA2RGB)
        cv2.imwrite("debug_frame.jpeg", rgb)

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
                agent_transform = (
                    self._default_agent.scene_node.transformation
                )
                object_transform = np.dot(agent_transform, self.grip_offset)
                object_type = self.get_object_type(gripped_object_id)

                if object_type == "art":
                    self.set_articulated_object_root_state(gripped_object_id, object_transform)
                    position = self.get_articulated_object_root_state(gripped_object_id).translation
                else:
                    self.set_transformation(object_transform, gripped_object_id)
                    position = self.get_translation(gripped_object_id)

                if self.pathfinder.is_navigable(position):
                    if object_type == "art":
                        self.set_articulated_object_motion_type(gripped_object_id, MotionType.STATIC)
                    else:
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
                    ids_dict = self.get_both_existing_object_ids()
                    agent_transform = self._default_agent.scene_node.transformation

                    if gripped_object_id in ids_dict["art"]:
                        obj_transform = self.get_articulated_object_root_state(gripped_object_id)
                        self.set_articulated_object_motion_type(
                            gripped_object_id, MotionType.KINEMATIC
                        )
                    else:
                        obj_transform = self.get_transformation(gripped_object_id)
                        self.set_object_motion_type(
                            MotionType.KINEMATIC, gripped_object_id
                        )

                    # (YK): This doesn't sum after dot-product
                    self.grip_offset = np.dot(
                        np.array(agent_transform.inverted()),
                        np.array(obj_transform),
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
        existing_object_ids_dict = self._sim.get_both_existing_object_ids()
        existing_object_ids = existing_object_ids_dict["art"] + existing_object_ids_dict["non_art"]
        return spaces.Discrete(len(existing_object_ids))

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

        object_ids = self._sim.get_both_existing_object_ids()
        if len(object_ids["non_art"]) > 0:
            object_id = object_ids["non_art"][0]
            object_position = self._sim.get_translation(object_id)
        elif len(object_ids["art"]) > 0:
            object_id = object_ids["art"][0]
            object_transform = self._sim.get_articulated_object_root_state(object_id)
            object_position = np.array(object_transform.translation)
        else:
            raise AssertionError

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


