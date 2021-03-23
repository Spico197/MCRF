import os
from functools import partial

from mcrf.utils.config import YamlConfig
from mcrf.tasks import NERTask
from mcrf.helper import load_lines
from mcrf.helper.manager import CachedManager
from mcrf.helper.collate_function import weibo_collate_fn


if __name__ == "__main__":
    config = YamlConfig("config.yaml")
    collate_fn = partial(weibo_collate_fn, config.device)
    data_manager = CachedManager(config, load_lines, collate_fn)

    setattr(config, 'num_tags', data_manager.transform.tag_lbe.num_tags)
    setattr(config, 'vocab_size', len(data_manager.transform.vocab))

    task = NERTask(config, data_manager)
    task.train()

    task.load(os.path.join(config.output_dir, "ckpt", "LSTMCRFModel.best.pth"))
    results = task.predict(["佟湘玉和李大嘴都在同福客栈工作。"])
    print(results)
