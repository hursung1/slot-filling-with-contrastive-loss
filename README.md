# Cross Domain Slot Filling with Contrastive Loss

* Data: [SNIPS Dataset](https://github.com/sonos/nlu-benchmark) preprocessed by [zliucr](https://github.com/zliucr/coach)
* Download data from [here](https://drive.google.com/drive/folders/1ydalMtB-hpfS3SIEaR5UbRfEe2m8bFcj) and save it in <b>./data</b> diretory.

## How to train & test(prediction)
* run code below
```python
python main.py config.json
```

## Options
mostly similar as options used in huggingface arguments
* --epoch: training epochs
* --tgt_dm: target domain, source domain consists of domains without except target domain
* --batch_size: batch size
* --lr: learning rate
* --dropout: dropout rate
* --n_samples: Used in training step. The number of samples for <i>few shot learning</i>. <b>Include n_samples number of target domain in source domain</b>.
* --test_mode: Used in test step. Set test mode as "<b>testset</b>" for <i>run_test_total.sh</i> or "<b>seen_unseen</b>" for <i>run_test_seen_unseen.sh</i>
