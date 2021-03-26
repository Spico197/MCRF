import math
from functools import partial
from typing import List

import torch
from tqdm import tqdm

from mcrf import modules
from mcrf import metrics
from mcrf.helper import get_entities_from_tag_seq, get_num_illegal_tags_from_tag_seq
from .base_task import BaseTask


class NERTask(BaseTask):
    def __init__(self, config, data_manager) -> None:
        super().__init__(config)
        self.data_manager = data_manager

        if self.config.crf_type == "PlainCRF":
            crf_class = getattr(modules, self.config.crf_type)
        elif self.config.crf_type == "ConstraintCRF":
            crf_class = getattr(modules, self.config.crf_type)
            crf_class = partial(
                crf_class, constraints=modules.allowed_transitions(
                    self.config.constraint_type, self.data_manager.transform.tag_lbe.id2label))
        elif self.config.crf_type == "MaskedCRF":
            crf_class = getattr(modules, self.config.crf_type)
            crf_class = partial(
                crf_class,
                constraints=modules.allowed_transitions(
                    self.config.constraint_type, self.data_manager.transform.tag_lbe.id2label),
                masked_training=self.config.masked_training,
                masked_decoding=self.config.masked_decoding)
        else:
            raise NotImplementedError
        crf = crf_class(self.config.num_tags)
        self.model = self.model_class(config, crf)
        self.mdoel = self.model.to(self.config.device)
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.config.lr)

        self.history = {"dev": [], "test": []}
        self.best_f1 = -math.inf
        self.best_epoch = -1

    def train(self):
        for epoch_idx in range(1, self.config.num_epoch + 1, 1):
            self.logging(f"Epoch: {epoch_idx}/{self.config.num_epoch}")
            self.model.train()
            loader = tqdm(self.data_manager.train_loader, desc="Train", ncols=80, ascii=True)
            for batch in loader:
                loss = self.model(**batch)
                loader.set_postfix({"loss": loss.item()})
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            measures = self.eval("dev")
            self.history['dev'].append(measures)
            test_measures = self.eval("test")
            self.history['test'].append(test_measures)

            is_best = False
            if measures['f1'] > self.best_f1:
                is_best = True
                self.best_f1 = measures['f1']
                self.best_epoch = epoch_idx

            if is_best and self.config.save_best_ckpt:
                self.save_ckpt("best", epoch_idx)

            self.logging(f"Epoch: {epoch_idx}, is_best: {is_best}, Dev: {measures}, Test: {test_measures}")

        self.logging((f"Trial finished. Best Epoch: {self.best_epoch}, Dev: {self.history['dev'][self.best_epoch - 1]}, "
                      f"Best Test: {self.history['test'][self.best_epoch - 1]}"))

    @torch.no_grad()
    def eval(self, dataset_name):
        self.model.eval()
        name2loader = {
            "train": self.data_manager.train_loader,
            "dev": self.data_manager.dev_loader,
            "test": self.data_manager.test_loader,
        }
        loader = tqdm(name2loader[dataset_name], desc=f"{dataset_name} Eval", ncols=80, ascii=True)
        seqs = []
        tags = []
        preds = []
        for batch in loader:
            tags.extend(batch['tags'].detach().cpu().tolist())
            del batch['tags']
            seqs.extend(batch['inputs'].detach().cpu().tolist())
            out = self.model(**batch)
            preds.extend(out)

        gold_ents_all = []
        pred_ents_all = []
        # num_illegal_tag = 0
        for seq, tag, pred in zip(seqs, tags, preds):
            seq = self.data_manager.transform.vocab.convert_ids_to_tokens(seq)
            tag = self.data_manager.transform.tag_lbe.decode(tag)
            pred = self.data_manager.transform.tag_lbe.decode(pred)
            # num_illegal_tag += get_num_illegal_tags_from_tag_seq(pred)
            gold_ents = get_entities_from_tag_seq(seq, tag)
            pred_ents = get_entities_from_tag_seq(seq, pred)
            gold_ents_all.append(gold_ents)
            pred_ents_all.append(pred_ents)

        measures = metrics.prf1_for_tagging(gold_ents_all, pred_ents_all)
        return measures["micro"]["overall"]

    @torch.no_grad()
    def predict(self, strings: List[str]):
        self.model.eval()
        batch = self.data_manager.transform.predict_transform(strings)
        tensor_batch = self.data_manager.collate_fn(batch)
        if "tags" in tensor_batch:
            del tensor_batch["tags"]
        outs = self.model(**tensor_batch)
        pred_ents_per_sent = []
        for string, pred_tag in zip(strings, outs):
            # in case of OOV
            refreshed_pred_ents = []
            pred_tag = self.data_manager.transform.tag_lbe.decode(pred_tag)
            pred_ents = get_entities_from_tag_seq(string, pred_tag)
            for ent_type in pred_ents:
                for ent in pred_ents[ent_type]:
                    refreshed_pred_ents.append((string[ent[2]:ent[3]], ent[1], ent[2], ent[3]))
            pred_ents_per_sent.append(refreshed_pred_ents)
        return pred_ents_per_sent
