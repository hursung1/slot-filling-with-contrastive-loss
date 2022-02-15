import os
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from itertools import chain, repeat

def repeater(dataloader): # for infinite dataloader loop
    for loader in repeat(dataloader):
        for data in loader:
            yield data

def train(model, 
            model_save_path,
            dataloader_train, 
            dataloader_val, 
            optim, 
            scheduler, 
            eval_steps, 
            total_steps, 
            early_stopping_patience,
            loss_key_ratio,
            log_dict
            ):
    """
    trainer function

    
    """
    model.train()
    log_dict['eval_results'] = []

    repeat_dataloader = repeater(dataloader_train) # for infinite loop
    pbar = tqdm(repeat_dataloader, total=total_steps, desc="Start Training")
    
    best_step = 0
    best_f1_score = 0
    patience = 0
    losses = []

    for i, features in enumerate(pbar):
        if i == total_steps:
            print(f"Training step reached set maximum steps: {total_steps}")
            break

        # features = {"utter": utter_data, "template & augdata": sth_data}
        query_input = features['utter']
        key_input = features['tem_aug_data']
        optim.zero_grad()

        _crf_loss, logits, _cl_loss = model(query_input, key_input)
        
        crf_loss = _crf_loss.mean()
        cl_loss = _cl_loss.mean()

        loss = crf_loss * (1 - loss_key_ratio) + cl_loss * loss_key_ratio
        loss.backward()
        losses.append(loss.detach().cpu().item())

        optim.step()
        scheduler.step()

        pbar.set_description(f"LOSS: {losses[-1]:.4f}")

        # evaluation
        if (i + 1) % eval_steps == 0:
            results = eval(model, dataloader_val)
            print(f"Results at step {i+1}")
            print(results)
            results['step'] = i + 1
            log_dict['eval_results'].append(results)

            eval_f1 = results['macro avg']['f1-score']
            if eval_f1 > best_f1_score:
                """
                when better evaluation f1 score is found:
                update best_f1_score and best_step
                & save model's parameter
                """
                print("Found better model!")

                os.makedirs(model_save_path, exist_ok=True)
                if os.path.isfile(model_save_path+f'best-model-parameters-step-{best_step+1}.pt'):
                    os.remove(model_save_path+f'best-model-parameters-step-{best_step+1}.pt')
                torch.save(model.state_dict(), model_save_path+f'best-model-parameters-step-{i+1}.pt')
                best_f1_score = eval_f1
                best_step = i
                patience = 0

            else:
                patience += 1
                if patience == early_stopping_patience:
                    print(f"Early stop at step {i+1}")
                    i += 1
                    break

            model.train()

    log_dict['stopped_step'] = i
    log_dict['eval_best_step'] = best_step
    log_dict['eval_best_f1_score'] = best_f1_score

    return best_step, best_f1_score

def eval(model, dataloader, file=None):
    """
    evalutation function for validation dataset and test dataset
    
    returns
    ----------
    List of losses, F1 Score
    """
    model.eval()
    losses = []
    total_preds = []
    total_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for features in pbar:
            # features = {"utter": utter_data, "template & augdata": sth_data}
            query_input = features['utter']
            # key_input = features['tem_aug_data']
            _loss, logits, _ = model(query_input)
            
            loss = _loss.mean()
            losses.append(loss.detach().cpu().item())
            
            pred = torch.argmax(logits, dim=2)
            true_labels = query_input['labels']
            
            total_preds.extend(pred.tolist())
            total_targets.extend(true_labels.tolist())
        
        # below is for check
        # rand = torch.randint(query.size()[0], (1,)).item()
        # decoded = tokenizer.decode(query[rand])

        # print("Query     : ", decoded)
        # print("Answer    : ", targets[rand])
        # print("Prediction: ", pred[rand])
        total_targets = list(chain.from_iterable(total_targets))
        total_preds = list(chain.from_iterable(total_preds))
        results = classification_report(total_targets, total_preds, output_dict=True)

    return results