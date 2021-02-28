#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_cos_eor_dataset():
    try:
        from habitat.datasets.cos_eor.cos_eor_dataset import (  # noqa: F401 isort:skip
            CosRearrangementDatasetV0,
        )
    except ImportError as e:
        cos_eor_import_error = e

        @registry.register_dataset(name="CosRearrangementDataset-v0")
        class CosRearrangementDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise cos_eor_import_error
