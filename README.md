# Cross Domain Slot Filling with Contrastive Loss

* I refered the code of [Coach](https://github.com/zliucr/coach)
* Data: [SNIPS Dataset](https://github.com/sonos/nlu-benchmark) preprocessed by [zliucr](https://github.com/zliucr/coach)
* Download data from [here](https://drive.google.com/drive/folders/1ydalMtB-hpfS3SIEaR5UbRfEe2m8bFcj) and save it in <b>./data</b> diretory.

## Model Configuration
Follows default config of pretrained BERT provided by *[huggingface transformers](https://huggingface.co/)*.

## How to run
* run code below
```python
python main.py config.json
```

## Options
Mostly similar as used in huggingface arguments.
To add option, you can modify **config.json** file.
You can get explanation of each options in **config.py**.
Below are mainly used options when running code.

* target_domain: The domain used when test. Training data consists of other domains.
* n_samples: # of data from target domain for *few shot learning*. Model conducts training under *zero-shot setting* when 0.
* learning_rate: learning rate
* dropout_rate: Dropout rate for BERT output hidden.
* max_steps: Maximum learning steps in terms of batch.
* eval_steps: Conduct evaluation for each (eval_steps) steps.
* early_stopping_patience: Patience steps after detecting the best model for early stopping.
* run_mode: train mode / test mode
