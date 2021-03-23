from typing import Dict, Optional, List, Union


class LabelEncoder(object):
    def __init__(self, initial_dict: Optional[Dict] = {}) -> None:
        self.label2id = initial_dict
        self.id2label = {idx: label for idx, label in enumerate(self.label2id)}

    def add(self, label: Union[str, int]):
        if label not in self.label2id:
            label_id = len(self.label2id)
            self.label2id[label] = label_id
            self.id2label[label_id] = label

    def encode(self, labels: List[Union[str, int]]):
        if not all(label in self.label2id for label in labels):
            raise ValueError("Not all label are in this encoder")
        return list(map(lambda x: self.label2id[x], labels))

    def decode(self, ids: List[int]):
        if not all(idx in self.id2label for idx in ids):
            raise ValueError("Not all idx are in this encoder")
        return list(map(lambda x: self.id2label[x], ids))

    def __len__(self):
        return len(self.label2id)

    @property
    def num_tags(self):
        return len(self)
