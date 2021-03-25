# Masked CRF

**NOT Official** Pytorch implemented Masked CRF.

## Quick Start

### Settings

All the settings are in `config.yaml`, you can change model settings from this file.

### Run

```bash
python run.py
```

## Results

Micro-F1 results

### WeiboNER

| Batch Size | Optimizer | Learning Rate | Max Seqence Length |
| ---------: | :-------: | ------------: | -----------------: |
|          8 |    SGD    |         0.015 |                128 |

| Method                      |    Dev |   Test |
| :-------------------------- | -----: | -----: |
| PlainCRF                    | 57.564 | 51.733 |
| MaskedCRF ( decoding only ) | 55.662 | 51.351 |
| MaskedCRF                   | 55.380 | 50.287 |


### MSRA

| Batch Size | Optimizer | Learning Rate | Max Seqence Length |
| ---------: | :-------: | ------------: | -----------------: |
|          8 |    SGD    |         0.015 |                250 |

Model is selected directly from test set since there is no official dev set.

| Method                      |  Dev | Test |
| :-------------------------- | ---: | ---: |
| PlainCRF                    |      |      |
| MaskedCRF ( decoding only ) |      |      |
| MaskedCRF                   |      |      |


## Acknowledgements

- Official: [MaskedCRF](https://github.com/DandyQi/MaskedCRF)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
- [allennlp](https://github.com/allenai/allennlp)
- [ChineseWeiboNER](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/)
