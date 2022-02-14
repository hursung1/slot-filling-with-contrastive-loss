import torch
from torch import nn

from transformers import AutoTokenizer, AdamW, HfArgumentParser, TrainingArguments, get_linear_schedule_with_warmup

from src.dataloader import get_dataloader
from src.model import SlotFillingModel
from config import DataTrainingArguments, ModelArguments
from trainer import train, eval

import sys
import os

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set default device
    cuda_available = True if torch.cuda.is_available() else False

    # load pretrained BERT and define model 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    model = SlotFillingModel(model_args)
    logfile_dir_path = f'{training_args.output_dir}/{data_args.target_domain}/Sample{data_args.n_samples}/'
    model_save_path = f'{training_args.output_dir}/model/{data_args.target_domain}/Sample{data_args.n_samples}/'
    
    if cuda_available:
        model = nn.DataParallel(model.cuda())

    # get dataloader
    dataloader_train, dataloader_val, dataloader_test = get_dataloader(
                                            data_args.target_domain, 
                                            data_args.n_samples, 
                                            data_args.dataset_path, 
                                            model_args.num_aug_data, 
                                            training_args.per_device_train_batch_size, 
                                            data_args.max_train_samples, 
                                            tokenizer)

    # loss function, optimizer, ...
    optim = AdamW(model.parameters(), lr=training_args.learning_rate, correct_bias=True)

    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=training_args.max_steps)

    os.makedirs(logfile_dir_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    logfile = open(logfile_dir_path + f'logfile_{data_args.target_domain}_sample{data_args.n_samples}.txt', 'w')
    print(f'Target Domain: {data_args.target_domain}\tN Samples: {data_args.n_samples}')
    logfile.write(f'Target Domain: {data_args.target_domain}\tN Samples: {data_args.n_samples}\n')

    best_step, best_f1 = train(model=model, 
                                dataloader_train=dataloader_train, 
                                dataloader_val=dataloader_val, 
                                optim=optim, 
                                scheduler=scheduler, 
                                eval_steps=training_args.eval_steps,
                                total_steps=training_args.max_steps,
                                early_stopping_patience=data_args.early_stopping_patience,
                                model_save_path=model_save_path,
                                loss_key_ratio=model_args.loss_key_ratio)

    print("Training finished.")
    print(f"Best validation f1 score {best_f1: .2f} at training step {best_step}")
    
    # Prediction / Test
    results = eval(model, dataloader_test)
    print("Prediction")
    print(results)
    print(f"F1-score {results['macro avg']['f1-score']: .2f} at prediction.")


    logfile.close()


if __name__=="__main__":
    main()