import os
from typing import Optional

import yaml
import torch

from . import set_seed
from .logging import Logger


class YamlConfig(object):
    r"""Easier utility class for configuration.

    Args:
        config_filepath: filepath to `YAML` config file
        make_taskdir: whether to make new sub-folder under `output_dir`.
            useful to set False when you load a checkpoint to predict
            something with its dumped config file.
    """
    def __init__(self, config_filepath: str, make_taskdir: Optional[bool] = True):
        self.config_abs_path = os.path.abspath(config_filepath)

        # do not use `with` expression here in order to find errors ealier
        fin = open(self.config_abs_path, 'rt', encoding='utf-8')
        __config = yaml.safe_load(fin)
        fin.close()

        for key, val in __config.items():
            setattr(self, key, val)

        if make_taskdir:
            self.output_dir = os.path.join(self.output_dir, self.task_name)
        __config.update({"output_dir": self.output_dir})
        self._config = __config

        overwritten_flag = False
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            overwritten_flag = True

        self.logger = Logger(
            self.task_name, log_path=os.path.join(self.output_dir, "log.log"))

        if overwritten_flag:
            self.logger.warning("Overwrite output directory.")

        with open(os.path.join(self.output_dir, "config.yaml"), 'wt', encoding='utf-8') as fout:
            yaml.dump(__config, fout)

        self.device = torch.device(self.device)

        set_seed(self.random_seed)

    def __str__(self):
        return str(self.__dict__)
