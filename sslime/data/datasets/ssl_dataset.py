#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from torch.utils.data import Dataset

from sslime.core.config import config as cfg
from sslime.data.datasets import DATASET_SOURCE_MAP
from sslime.data.ssl_transforms import get_transform


class GenericSSLDataset(Dataset):
    """Base Self Supervised Learning Dataset Class."""

    def __init__(self, split):
        self.split = split
        assert len(cfg[split].DATA_SOURCES) == len(
            cfg[split].DATA_PATHS
        ), "Mismatch between length of data_sources and data paths provided"
        self.data_objs = []
        self.label_objs = []
        for source, path in zip(cfg[split].DATA_SOURCES, cfg[split].DATA_PATHS):
            self.data_objs.append(DATASET_SOURCE_MAP[source](path, split))

        if cfg[split].LABEL_PATHS is not None:
            assert len(cfg[split].LABEL_SOURCES) == len(
                cfg[split].LABEL_PATHS
            ), "Mismatch between length of label_sources and label paths provided"
            for source, path in zip(cfg[split].LABEL_SOURCES, cfg[split].LABEL_PATHS):
                # For now, only disk source is supported for labels.
                # Others may be supported later on.
                assert source == "disk", "Other sources not supported yet."
                if cfg[split].MMAP_MODE:
                    label_file = np.load(path, mmap_mode="r")
                else:
                    label_file = np.load(path)
                self.label_objs.append(label_file)

        self.transform = get_transform(cfg[split].TRANSFORMS)

    def __getitem__(self, idx):
        item = {"data": [], "data_valid": []}
        for source in self.data_objs:
            data, valid = source[idx]
            item["data"].append(data)
            item["data_valid"].append(valid)

        if self.label_objs:
            item["label"] = []
            for source in self.label_objs:
                item["label"].append(source[idx])

        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.data_objs[0])

    def num_samples(self, source_idx):
        return len(self.data_objs[source_idx])
