#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_cos_eor_task():
    try:
        from habitat.tasks.cos_eor.cos_eor import CosRearrangementTask  # noqa: F401
    except ImportError as e:
        cos_eor_task_import_error = e

        @registry.register_task(name="CosRearrangementTask-v0")
        class CosRearrangementTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise cos_eor_task_import_error
