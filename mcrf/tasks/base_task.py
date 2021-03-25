import os
import json
import logging
from typing import Optional

import torch
from torch import optim
from torch import distributed
from torch.nn import parallel

from mcrf import models
from mcrf.helper import transform


class BaseTask(object):
    def __init__(self, config) -> None:
        self.config = config
        self.model_class = getattr(models, self.config.model)
        self.optimizer_class = getattr(optim, self.config.optimizer)
        self.model = None
        self.optimizer = None

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, path: str,
             load_model: Optional[bool] = True,
             load_optimizer: Optional[bool] = False,
             load_data_manager: Optional[bool] = True):
        if os.path.exists(path):
            self.logging('Resume checkpoint from {}'.format(path))
        else:
            raise ValueError('Checkpoint does not exist, {}'.format(path))

        if torch.cuda.device_count() == 0:
            store_dict = torch.load(path, map_location='cpu')
        else:
            store_dict = torch.load(path, map_location=self.config.device)

        self.logging('Setting: {}'.format(
            json.dumps(store_dict['setting'], ensure_ascii=False, indent=2)
        ))

        if load_model:
            if self.model and 'model_state' in store_dict:
                if isinstance(self.model, parallel.DataParallel) or \
                        isinstance(self.model, parallel.DistributedDataParallel):
                    self.model.module.load_state_dict(store_dict['model_state'])
                else:
                    self.model.load_state_dict(store_dict['model_state'])
                self.logging('Load model successfully')
            else:
                raise ValueError(f"Model loading failed. self.model={self.model}, stored_dict_keys={store_dict.keys()}")
        else:
            self.logging("Not load model")

        if load_optimizer:
            if self.optimizer and 'optimizer_state' in store_dict:
                self.optimizer.load_state_dict(store_dict['optimizer_state'])
                self.logging('Load optimizer successfully')
            else:
                raise ValueError(f"Model loading failed. self.model={self.optimizer}, stored_dict_keys={store_dict.keys()}")
        else:
            self.logging("Not load optimizer")

        if load_data_manager:
            self.data_manager = store_dict['data_manager']

    def save(self, path, epoch: Optional[int] = None):
        self.logging(f"Dumping checkpoint into: {path}")
        store_dict = {
            'setting': self.config._config,
        }

        if self.model:
            if isinstance(self.model, parallel.DataParallel) or \
                    isinstance(self.model, parallel.DistributedDataParallel):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            store_dict['model_state'] = model_state
        else:
            self.logging('No model state is dumped', logging.WARNING)

        if self.optimizer:
            store_dict['optimizer_state'] = self.optimizer.state_dict()
        else:
            self.logging('No optimizer state is dumped', logging.WARNING)

        if epoch:
            store_dict['epoch'] = epoch

        if self.data_manager is not None:
            store_dict['data_manager'] = self.data_manager

        torch.save(store_dict, path)

    def save_ckpt(self, identifier, epoch: Optional[int] = None):
        ckpt_dir = os.path.join(self.config.output_dir, "ckpt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_name = f"{self.model.__class__.__name__}.{identifier}.pth"
        self.save(os.path.join(ckpt_dir, ckpt_name), epoch)

    def logging(self, msg: str, level: Optional[int] = logging.INFO):
        if self.in_distributed_mode():
            msg = 'Rank {} {}'.format(distributed.get_rank(), msg)
        if self.config.only_master_logging:
            if self.is_master_node():
                self.config.logger.logger.log(level, msg)
        else:
            self.config.logger.logger.log(level, msg)

    def is_master_node(self):
        if self.in_distributed_mode():
            if distributed.get_rank() == 0:
                return True
            else:
                return False
        else:
            return True

    def in_distributed_mode(self):
        return self.config.local_rank >= 0
