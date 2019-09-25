#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

from sslime.core.config import cfg_from_file, cfg_from_list
from sslime.workflows.generic_trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL model training")
    parser.add_argument(
        "--config_file", type=str, required=True, help="Config file for params"
    )
    parser.add_argument(
        "opts",
        help="see config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)

    Trainer().train()
