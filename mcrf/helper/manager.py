from torch.utils.data import DataLoader

from mcrf.helper import transform

from .dataset import CachedDataset


class CachedManager(object):
    def __init__(self, config, load_fn, collate_fn, **kwargs) -> None:
        self.config = config
        self.collate_fn = collate_fn

        transform_class = getattr(transform, self.config.transform)
        self.transform = transform_class(**kwargs.get("transform", {"max_seq_len": self.config.max_seq_len}))

        self.train_set = CachedDataset(self.transform(load_fn(self.config.train_filepath)))
        self.dev_set = CachedDataset(self.transform(load_fn(self.config.dev_filepath)))
        self.test_set = CachedDataset(self.transform(load_fn(self.config.test_filepath)))

        self.train_loader = DataLoader(self.train_set, batch_size=self.config.train_batch_size, shuffle=True, collate_fn=collate_fn)
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.config.eval_batch_size, shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=self.config.eval_batch_size, shuffle=False, collate_fn=collate_fn)
