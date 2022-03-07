import torch
from torch import nn

from transformers import AutoTokenizer, AdamW, HfArgumentParser, TrainingArguments, get_linear_schedule_with_warmup

from src.dataloader import get_dataloader
from src.model import SlotFillingModel
from config import DataTrainingArguments, ModelArguments
from trainer import train, eval
from src.utils import log_params

import sys
import os
import time
import json

def main():
    # # set visible gpus
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # parse arguments 
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # For logging
    current_time = time.localtime()
    current_time = f"{current_time.tm_year}_{current_time.tm_mon}_{current_time.tm_mday}_{current_time.tm_hour}_{current_time.tm_min}_{current_time.tm_sec}"
    
    log_path = f'{training_args.output_dir}/{data_args.target_domain}/Sample{data_args.n_samples}/{current_time}/'
    model_path = f'{training_args.output_dir}/model/{data_args.target_domain}/Sample{data_args.n_samples}/{current_time}/'
    log_dict = {}
    
    log_params(log_dict, [model_args, data_args, training_args])

    # load pretrained BERT and define model 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    model = nn.DataParallel(SlotFillingModel(model_args).cuda()) if torch.cuda.is_available() else SlotFillingModel(model_args)
    
    # get dataloader
    dataloader_train, dataloader_val, dataloader_test = get_dataloader(
                                            data_args.target_domain, 
                                            data_args.n_samples, 
                                            data_args.dataset_path, 
                                            model_args.key_enc_data,
                                            model_args.num_key_enc_data,
                                            training_args.per_device_train_batch_size, 
                                            data_args.max_train_samples, 
                                            tokenizer,
                                            model_args.use_both_or_not,
                                            model_args.num_aug)

    # loss function, optimizer, ...
    optim = AdamW(model.parameters(), lr=training_args.learning_rate, correct_bias=True)

    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps)

    os.makedirs(model_path, exist_ok=True)
    
    print(f'Target Domain: {data_args.target_domain}\tN Samples: {data_args.n_samples}')

    best_step, best_f1 = train(model=model, 
                                dataloader_train=dataloader_train, 
                                dataloader_val=dataloader_val, 
                                optim=optim, 
                                scheduler=scheduler, 
                                eval_steps=training_args.eval_steps,
                                total_steps=training_args.max_steps,
                                early_stopping_patience=data_args.early_stopping_patience,
                                model_save_path=model_path,
                                loss_key_ratio=model_args.loss_key_ratio,
                                log_dict=log_dict)

    print("Training finished.")
    print(f"Best validation f1 score {best_f1: .2f} at training step {best_step}")
    
    # Prediction / Test
    model.load_state_dict(torch.load(model_path+f"best-model-parameters-step-{best_step+1}.pt"))
    results = eval(model, dataloader_test)
    print(f"F1 Score at prediction: {results['fb1']}")

    log_dict['test_result'] = results['fb1']
    
    os.makedirs(log_path, exist_ok=True)
    with open(log_path + 'log.json', 'w') as json_out:
        json.dump(log_dict, json_out, indent=4)

    return 


if __name__=="__main__":
    main()