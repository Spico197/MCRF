import copy
from typing import Iterable, Optional, Any, List

from .vocab import Vocab
from .label_encoder import LabelEncoder


class BaseTransform(object):
    def __init__(self, max_seq_len) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        self.vocab = Vocab()
        self.tag_lbe = LabelEncoder({'O': 0})

    def transform(self, lines: Iterable):
        raise NotImplementedError

    def predict_transform(self, strings: Iterable[str]):
        ret_data = []
        for string in strings:
            seq_len = min(len(string), self.max_seq_len)
            comp_len = max(0, (self.max_seq_len - seq_len))
            ret_data.append({
                "inputs": self.vocab.convert_tokens_to_ids(list(string))[:seq_len] + comp_len * [self.vocab.pad_idx],
                "mask": seq_len * [1] + comp_len * [0],
            })
        return ret_data

    def __call__(self, *args: Any, **kwargs: Any):
        return self.transform(*args, **kwargs)


class CachedCharSegTSVTransform(BaseTransform):
    def __init__(self, max_seq_len, sep: Optional[str] = '\t') -> None:
        super().__init__(max_seq_len)
        self.sep = sep
        self.seg_lbe = LabelEncoder()

    def raw_transform(self, lines_of_data: Iterable):
        data = []
        chars = []
        segs = []
        tags = []
        for line in lines_of_data:
            line = line.strip().split(self.sep)
            if len(line) != 2:
                # end of sentence
                if len(chars) > 0:
                    data.append({
                        "char": copy.deepcopy(chars),
                        "seg": copy.deepcopy(segs),
                        "tag": copy.deepcopy(tags)
                    })
                    chars.clear()
                    segs.clear()
                    tags.clear()
                # else: start of doc or multiple CRLF after this line
                continue
            else:
                tag = line[1]
                try:
                    char, seg = line[0][0], int(line[0][1:])
                except ValueError:
                    char, seg = '[UNK]', 0
                chars.append(char)
                segs.append(seg)
                tags.append(tag)

                self.vocab.add(char)
                self.seg_lbe.add(seg)
                self.tag_lbe.add(tag)
        return data

    def transform(self, lines_of_data: Iterable):
        data = self.raw_transform(lines_of_data)

        ret_data = []
        for d in data:
            seq_len = min(len(d['char']), self.max_seq_len)
            comp_len = max(0, (self.max_seq_len - seq_len))
            ret_data.append({
                "inputs": self.vocab.convert_tokens_to_ids(d['char'])[:seq_len] + comp_len * [self.vocab.pad_idx],
                "mask": seq_len * [1] + comp_len * [0],
                "segs": self.seg_lbe.encode(d['seg'])[:seq_len] + comp_len * [0],
                "tags": self.tag_lbe.encode(d['tag'])[:seq_len] + comp_len * [self.tag_lbe.label2id['O']],
            })
        return ret_data


class CachedCharTSVTransform(BaseTransform):
    def __init__(self, max_seq_len, sep: Optional[str] = '\t') -> None:
        super().__init__(max_seq_len)
        self.sep = sep

    def raw_transform(self, lines_of_data: Iterable):
        data = []
        chars = []
        tags = []
        for line in lines_of_data:
            line = line.strip().split(self.sep)
            if len(line) != 2:
                # end of sentence
                if len(chars) > 0:
                    data.append({
                        "char": copy.deepcopy(chars),
                        "tag": copy.deepcopy(tags)
                    })
                    chars.clear()
                    tags.clear()
                # else: start of doc or multiple CRLF after this line
                continue
            else:
                char = line[0]
                tag = line[1]
                chars.append(char)
                tags.append(tag)

                self.vocab.add(char)
                self.tag_lbe.add(tag)
        return data

    def transform(self, lines_of_data: Iterable):
        data = self.raw_transform(lines_of_data)

        ret_data = []
        for d in data:
            seq_len = min(len(d['char']), self.max_seq_len)
            comp_len = max(0, (self.max_seq_len - seq_len))
            ret_data.append({
                "inputs": self.vocab.convert_tokens_to_ids(d['char'])[:seq_len] + comp_len * [self.vocab.pad_idx],
                "mask": seq_len * [1] + comp_len * [0],
                "tags": self.tag_lbe.encode(d['tag'])[:seq_len] + comp_len * [self.tag_lbe.label2id['O']],
            })
        return ret_data
