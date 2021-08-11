<h1 style="text-align:center">Masked CRF</h1>

**NOT Official** Pytorch implemented Masked CRF.

## ⚙️Installation

### Dependencies
- Python >= 3.6
  - torch == 1.5.1 (better > 1.0)
  - tqdm == 4.53.0
  - pyyaml == 5.3.1

### Download and Install

1. Editable installation:

```bash
pip install -e .
```

2. Install from PyPI:

```bash
pip install pytorch-mcrf -i https://pypi.org/simple
```

## 🚀Quick Start

### Settings

All the settings are in `config.yaml`, you can change model settings from this file.

### Run

```bash
python run.py
```

## 📝Results

Micro-F1 results

### WeiboNER

| Batch Size | Optimizer | Learning Rate | Max Seqence Length |
| ---------: | :-------: | ------------: | -----------------: |
|          8 |    SGD    |         0.015 |                128 |

| Method                      |    Dev |   Test | #Illegal Tags |
| :-------------------------- | -----: | -----: | ------------: |
| PlainCRF                    | 57.563 | 51.792 |             4 |
| MaskedCRF ( decoding only ) | 57.143 | 51.656 |             0 |
| MaskedCRF                   | 55.380 | 50.358 |             0 |


### MSRA

| Batch Size | Optimizer | Learning Rate | Max Seqence Length |
| ---------: | :-------: | ------------: | -----------------: |
|          8 |    SGD    |         0.015 |                250 |

Model is selected directly from test set since there is no official dev set.

| Method                      |   Test | #Illegal Tags |
| :-------------------------- | -----: | ------------: |
| PlainCRF                    | 85.163 |             0 |
| MaskedCRF ( decoding only ) | 85.163 |             0 |
| MaskedCRF                   | 84.693 |             0 |


## ❤️Acknowledgements

- Official: [MaskedCRF](https://github.com/DandyQi/MaskedCRF)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
- [allennlp](https://github.com/allenai/allennlp)
- [ChineseWeiboNER](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/)

## ✨UPDATES
- v0.0.3: fix an issue when recovering entities from tags
- v0.0.2: fix setuptools packages finding issue
