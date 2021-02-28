from typing import Type, Any

from habitat.tasks.nav.nav import NavigationTask, merge_sim_episode_config
from habitat.core.registry import registry
from habitat.config.default import Config
from habitat.core.dataset import Episode


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
