from typing import Type, Any

import numpy as np
from habitat.config.default import Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationTask, merge_sim_episode_config


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_to_goal_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        ids_pos = self._sim.get_both_existing_object_ids_with_positions()
        previous_position = ids_pos["art_pos"][0] if len(ids_pos["art_pos"]) > 0 else ids_pos["non_art_pos"][0]
        previous_position = np.array(previous_position).tolist()

        goal_position = episode.goals.position
        self._metric = self._euclidean_distance(
            previous_position, goal_position
        )


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_to_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        ids_pos = self._sim.get_both_existing_object_ids_with_positions()
        previous_position = ids_pos["art_pos"][0] if len(ids_pos["art_pos"]) > 0 else ids_pos["non_art_pos"][0]
        previous_position = np.array(previous_position).tolist()

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position

        self._metric = self._euclidean_distance(
            previous_position, agent_position
        )


def merge_sim_episode_with_object_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()
    sim_config.objects = [episode.objects.__dict__]
    sim_config.freeze()

    return sim_config


@registry.register_task(name="CosRearrangementTask-v0")
class CosRearrangementTask(NavigationTask):
    r"""Embodied Rearrangement Task
    Goal: An agent must place objects at their corresponding goal position.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)
