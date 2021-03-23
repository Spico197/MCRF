import copy
from typing import Iterable, Optional, Any


class MetaTransform(object):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def transform(cls, lines):
        raise NotImplementedError("Process is not implemented and should be override")

    def __call__(self, *args: Any, **kwargs: Any):
        return self.transform(*args, **kwargs)


class CachedCharSegTSV(MetaTransform):
    def __init__(self, sep: Optional[str] = '\t') -> None:
        super().__init__()
        self.sep = sep
    
    @classmethod
    def transform(cls, lines_of_data: Iterable, sep: Optional[str] = None):
        if sep:
            cls.sep = sep

        data = []
        chars = []
        segs = []
        tags = []
        for line in lines_of_data:
            line = line.strip().split(cls.sep)
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
                char, seg = line[0], int(line[1:])
                chars.append(char)
                segs.append(seg)
                tags.append(tag)
        return data
