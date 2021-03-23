from typing import Optional, List


class Vocab(object):
    def __init__(self, pad: Optional[str] = '[PAD]', unk: Optional[str] = "[UNK]") -> None:
        self.pad = pad
        self.unk = unk
        self.token2id = {pad: 0, unk: 1}
        self.pad_idx = self.token2id[pad]
        self.unk_idx = self.token2id[unk]

        self.id2token = {0: pad, 1: unk}

    def add(self, token: str):
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token

    def convert_tokens_to_ids(self, tokens: List[str]):
        return list(map(lambda x: self.token2id.get(x, self.unk_idx), tokens))

    def convert_ids_to_tokens(self, ids: List[int]):
        if not all(idx in self.id2token for idx in ids):
            raise ValueError("Not all idx are in this vocab")
        return list(map(lambda x: self.id2token[x], ids))

    def __len__(self):
        return len(self.token2id)


class FreqVocab(object):
    pass
