from habitat.core.registry import registry
from habitat.core.simulator import Simulator

# from habitat.sims.habitat_simulator.actions import (
#     HabitatSimV1ActionSpaceConfiguration,
# )


def _try_register_cos_eor_sim():
    try:
        import cos_eor_sim  # noqa: F401

        has_cos_eor_sim = True
    except ImportError as e:
        has_cos_eor_sim = False
        cos_eor_sim_import_error = e

    if has_cos_eor_sim:
        from habitat.sims.cos_eor.actions import (  # noqa: F401
            RearrangementSimV0ActionSpaceConfiguration,
        )
    else:

        @registry.register_simulator(name="CosRearrangementSim-v0")
        class CosRearrangementSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise cos_eor_sim_import_error
