import torch
from torch import nn

from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, HfArgumentParser, TrainingArguments, get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer

from src.utils import save_plot
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
    # print(params.use_plain)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    model = SlotFillingModel(model_args)
    # model = BertModel.from_pretrained('bert-base-uncased')
    logfile_dir_path = f'{training_args.output_dir}/{data_args.target_domain}/Sample{data_args.n_samples}/'
    model_save_path = f'{training_args.output_dir}/model/{data_args.target_domain}/Sample{data_args.n_samples}/'
    # bert_model_save_path = f'./experiments/Bert/Plain/{data_args.target_domain}/Sample{data_args.n_samples}/'
    # tagger_model_save_path = f'./experiments/Tagger/Plain/{data_args.target_domain}/Sample{data_args.n_samples}/'

    # BIOTagger = TripletTagger(model.config.hidden_size)
    
    if cuda_available:
        model = nn.DataParallel(model.cuda())
        # BIOTagger = BIOTagger.cuda()



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
    # model_parameters = [
    #     {"params": model.parameters()},
    #     {"params": BIOTagger.parameters()}
    # ]
    optim = AdamW(model.parameters(), lr=training_args.learning_rate, correct_bias=True)

    # total_steps = params.epoch * len(dataloader_tr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=training_args.max_steps)

    os.makedirs(logfile_dir_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    # os.makedirs(figure_dir_path, exist_ok=True)
    # os.makedirs(bert_model_save_path, exist_ok=True)
    # os.makedirs(tagger_model_save_path, exist_ok=True)
    
    logfile = open(logfile_dir_path + f'logfile_{data_args.target_domain}_sample{data_args.n_samples}.txt', 'w')
    print(f'Target Domain: {data_args.target_domain}\tN Samples: {data_args.n_samples}')
    logfile.write(f'Target Domain: {data_args.target_domain}\tN Samples: {data_args.n_samples}\n')

    # max_val_f1 = 0
    # best_model_counter = 0
    # e = 0
    
    # val_losses, val_acc, val_f1 = eval(model, BIOTagger, tokenizer, dataloader_val, cuda_available, False)
    # print(f"Before Training\nVal Loss {sum(val_losses)/len(val_losses):.4f}\tVal Accuracy {val_acc:.4f}\tF1 Score {val_f1:.4f}")
    # logfile.write(f"Before Training\nVal Loss {sum(val_losses)/len(val_losses):.4f}\tVal Accuracy {val_acc:.4f}\tF1 Score {val_f1:.4f}\n")

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

    #     # save model which shows best validation f1 score
    #     best_model_counter += 1
    #     e += 1
    #     if val_f1 > max_val_f1:
    #         print("Found Better Model!")
    #         logfile.write("Found Better Model!\n")
    #         tokenizer.save_pretrained(bert_model_save_path)
    #         model.module.save_pretrained(bert_model_save_path)
    #         torch.save(BIOTagger.state_dict(), tagger_model_save_path+'state_dict_model.pt')
    #         max_val_f1 = val_f1
    #         best_model_counter = 0


    logfile.close()


# def test(params):
#     # test for seen / unseen labeled data
#     cuda_available = False
#     if torch.cuda.is_available():
#         cuda_available = True

#     if params.use_plain:
#         bert_model_save_path = f'./experiments/Bert/Plain/{params.tgt_dm}/Sample{params.n_samples}/'
#         tagger_model_save_path = f'./experiments/Tagger/Plain/{params.tgt_dm}/Sample{params.n_samples}/state_dict_model.pt'
#         tokenizer = BertTokenizer.from_pretrained(bert_model_save_path)
#         model = BertModel.from_pretrained(bert_model_save_path)

#     else:
#         bert_model_save_path = f'./experiments/Bert/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/'
#         tagger_model_save_path = f'./experiments/Tagger/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/state_dict_model.pt'
#         tokenizer = AutoTokenizer.from_pretrained(bert_model_save_path)
#         model = AutoModelForTokenClassification.from_pretrained(bert_model_save_path)
        
#     BIOTagger = TripletTagger(model.config.hidden_size)
#     BIOTagger.load_state_dict(torch.load(tagger_model_save_path))

#     if cuda_available:
#         model = model.cuda()
#         BIOTagger = BIOTagger.cuda()

#     _, _, dataloader_test = get_dataloader(params.tgt_dm, params.batch_size, params.n_samples, tokenizer)

#     test_losses, test_acc, test_f1 = eval(model, BIOTagger, tokenizer, dataloader_test, cuda_available, True)
#     avg_test_loss = sum(test_losses)/len(test_losses)

#     print(f"Test\nLoss: {avg_test_loss:.4f}\tAccuracy: {test_acc:.4f}\tF1 Score: {test_f1:.4f}")
    

if __name__=="__main__":
    main()
    # if len(params.model_path) == 0:
    #     main(params)
    # else:
    #     test(params)