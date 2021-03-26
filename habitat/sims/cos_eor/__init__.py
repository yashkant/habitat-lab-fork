from habitat.core.registry import registry
from habitat.core.simulator import Simulator

# from habitat.sims.habitat_simulator.actions import (
#     HabitatSimV1ActionSpaceConfiguration,
# )


def _try_register_cos_eor_sim():
    try:
        import habitat.sims.cos_eor  # noqa: F401

        has_cos_eor_sim = True
    except ImportError as e:
        has_cos_eor_sim = False
        cos_eor_sim_import_error = e

    if has_cos_eor_sim:
        from habitat.sims.cos_eor.cos_eor import CosRearrangementSim  # noqa: F401
    else:

        @registry.register_simulator(name="CosRearrangementSim-v0")
        class CosRearrangementSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise cos_eor_sim_import_error
