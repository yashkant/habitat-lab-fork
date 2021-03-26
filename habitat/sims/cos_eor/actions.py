from typing import List, Any

import habitat_sim
from tqdm import tqdm

from habitat.core.embodied_task import SimulatorTaskAction
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat_sim.agent.controls.controls import ActuationSpec
import attr
import magnum as mn
from habitat.core.registry import registry


def raycast(sim, sensor_name, crosshair_pos=(128, 128), max_distance=2.0):
    r"""Cast a ray in the direction of crosshair and check if it collides
    with another object within a certain distance threshold
    :param sim: Simulator object
    :param sensor_name: name of the visual sensor to be used for raycasting
    :param crosshair_pos: 2D coordiante in the viewport towards which the
        ray will be cast
    :param max_distance: distance threshold beyond which objects won't
        be considered
    """
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera
    center_ray = render_camera.unproject(mn.Vector2i(crosshair_pos))
    raycast_results = sim.cast_ray(center_ray, max_distance=max_distance)

    # debug code
    # import cv2
    # obs = sim.get_sensor_observations()
    # rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGBA2RGB)
    # cv2.imwrite("before_hits.jpeg", rgb)
    # x, y = crosshair_pos[1], crosshair_pos[0]
    # rgb[x-1:x+1, y-1:y+1, :] = [255, 0, 0]
    # cv2.imwrite("current_pointer.jpeg", rgb)
    #
    # x_min, y_min, x_max, y_max = 1e3, 1e3, -1, -1
    # for x in tqdm(range(0, rgb.shape[0])):
    #     for y in range(0, rgb.shape[1]):
    #         center_ray = render_camera.unproject(mn.Vector2i([y, x]))
    #         raycast_results = sim.cast_ray(center_ray, max_distance=4.0)
    #         if raycast_results.has_hits() > 0:
    #             hit_ids = [obj.object_id >= 0 for obj in raycast_results.hits]
    #             if any(hit_ids):
    #
    #                 x_min = min(x_min, x)
    #                 y_min = min(y_min, y)
    #
    #                 x_max = max(x_max, x)
    #                 y_max = max(y_max, y)
    #
    #                 # (YK): Manipulated just like a 3D tensor, origin at top-left
    #                 #  down/right is dimension 1/2
    #                 # rgb[x-1:x+1, y-1:y+1, :] = [255, 0, 0]
    #
    #                 rgb[x, y, :] = [255, 0, 0]
    # cv2.imwrite("after_hits.jpeg", rgb)
    #
    # center_ray = render_camera.unproject(mn.Vector2i(crosshair_pos))
    # raycast_results = sim.cast_ray(center_ray, max_distance=max_distance)
    # print(f"mask h: {x_min}-{x_max}, w:{y_min}-{y_max}")

    closest_object = -1
    closest_dist = 1000.0
    if raycast_results.has_hits():
        for hit in raycast_results.hits:
            # Todo -- raycast actually returns non-existent object-ids,
            #  adding a hack to filter them, but needs to be figured out.
            #  it might be treating two surfaces as different objects? (idk)
            ids_dict = sim.get_both_existing_object_ids()
            ids = ids_dict["art"] + ids_dict["non_art"]

            # don't count non-existent hits
            if hit.ray_distance < closest_dist and hit.object_id in ids:
                closest_dist = hit.ray_distance
                closest_object = hit.object_id

    return closest_object


@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [128, 128]
    amount: float = 2.0


@registry.register_action_space_configuration(name="CosRearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    visual_sensor_name=self.config.VISUAL_SENSOR,
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            )
        }

        config.update(new_config)

        return config


@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        return self._sim.step(HabitatSimActions.GRAB_RELEASE)


