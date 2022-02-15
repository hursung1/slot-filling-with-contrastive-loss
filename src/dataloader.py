from src.datareader import PAD_INDEX, datareader
from src.utils import make_syn_data

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import random

class Dataset(data.Dataset):
    def __init__(self, tokenizer, utter, slot, domain, label, template_list=None, slot_exemplars=None, how_many_augs=3, label_pad_token_id=0):
        """
        how_many_augs: number of augmented data per aa utterance생성할 augmented data의 개수
        utter: original utterance
        slot: every slots
        label: b/i/o tag for each slot label
        template_list: templates
        slot_exemplars: dictionary, key=slot_label, value=list of exemplar words
        """
        self.utter = utter
        self.slot = slot
        self.domain = domain
        self.label = label
        self.template_list = template_list # this contains one positive template & two negative templates for each utter
        self.label_pad_token_id = label_pad_token_id
        if self.template_list is not None:
            self.slot_exemplars = slot_exemplars
            self.how_many_augs = how_many_augs
            self.mode = "train"
        else:
            self.mode = "eval"

        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = {}
        if self.mode == "train": # dataset mode "train": make synthetic data
            aug_data = make_syn_data(self.slot_exemplars, self.template_list[index], self.how_many_augs)
            tem_aug_emb = []
            prob = random.random()
            pos_data = []
            neg_data = []
            # remove T- from template && tokenize
            for i, template in enumerate(self.template_list[index]):
                tem_words = template.split()
                for j, word in enumerate(tem_words):
                    if "T-" in word:
                        tem_words[j] = tem_words[j][2:]
                
                tem = " ".join(tem_words)
                tem_emb = self.tokenizer(self.slot[index], tem)    
                if i == 0:
                    if aug_data is None:
                        pos_data.append(tem_emb)
                    if aug_data is not None and prob < 0.5:
                        pos_data.append(tem_emb)
                
                elif i > 0:
                    neg_data.append(tem_emb)
            
            # tokenize augmented data
            if aug_data is not None:
                for i, aug_datum in enumerate(aug_data):
                    aug_emb = self.tokenizer(self.slot[index], aug_datum)
                    if i == 0 and prob >= 0.5:
                        pos_data.append(aug_emb)
                    elif i > 0:
                        neg_data.append(aug_emb)

                # Combine template embeddings and augmented data embeddings
            tem_aug_emb.extend(pos_data)
            tem_aug_emb.extend(neg_data)
            
            item['tem_aug_data'] = tem_aug_emb

        else:
            item['tem_aug_data'] = None 

        # slot[SEP]utterance embedding
        slot_splits = self.slot[index].split()
        utter_splits = self.utter[index].split()
        slot_utter_emb = self.tokenizer(slot_splits, utter_splits, is_split_into_words=True) # input: slot_label <SEP> utter
        labels = self.label[index]

        # original utterance is tokenized: subword tokenization -> labels should be modified 
        tok_original_idx = slot_utter_emb.word_ids()
        # BIO label masking for [CLS] slot label [SEP]
        none_counter = 0
        new_labels = []
        label_mask = [] # for masking
        for i, word_idx in enumerate(tok_original_idx):
            if none_counter < 2 or word_idx is None:
                new_labels.append(self.label_pad_token_id)
                label_mask.append(0)
                if word_idx is None:
                    none_counter += 1
            
            elif none_counter == 2:
                new_labels.append(labels[word_idx])
                label_mask.append(1)

        slot_utter_emb['labels'] = new_labels
        slot_utter_emb['label_mask'] = label_mask
        item['utter'] = slot_utter_emb
        
        return item
    
    def __len__(self):
        return len(self.utter)


def pad_tensor(features, max_seq_len):
# def pad_tensor(dim0, dim1, tensor, front_pad=None):
    """
    features: list of lists, each element equals to output of hf tokenizer
    """
    padded_features = []
    for f in features:
        original_value_len = len(f)
        for _ in range(original_value_len, max_seq_len):
            f.append(PAD_INDEX)
        
        padded_features.append(f)

    return torch.tensor(padded_features, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(padded_features, dtype=torch.long)


def collate_fn(features):
    """
    Collate function for SLU Model
    pad at right side
    """
    utter_features = []
    tem_aug_features = []
    padded_features = {}

    for feature in features:
        utter_features.append(feature['utter'])
        try: 
            tem_aug_features.extend(feature['tem_aug_data'])
        except: # if feature['tem_aug_data'] is None
            tem_aug_features = None

    # this is for tem_aug_features' labels
    features_list = [utter_features, tem_aug_features]
    batch_feature_name = ["utter", "tem_aug_data"]

    batch_size = len(utter_features)
    try:
        tem_per_utter = len(features[0]["tem_aug_data"])
    except:
        pass
    max_seq_len = 0

    for key, _features in zip(batch_feature_name, features_list):
        if _features is None:
            continue

        _batch = {}
        for f in _features:
            for k, v in f.items():
                try:
                    _batch[k].append(v)
                except:
                    _batch[k] = [v]
            
                feature_len = len(v)
                if feature_len > max_seq_len:
                    max_seq_len = feature_len

        for k, v in _batch.items():
            v = pad_tensor(v, max_seq_len)
            v = v.reshape(batch_size, tem_per_utter, -1) if key == "tem_aug_data" else v
            _batch[k] = v

        padded_features[key] = _batch

    return padded_features


def get_dataloader(tgt_domain, n_samples, data_path, num_augs, batch_size, max_train_samples=-1, tokenizer=None):
    """
    input
    ----------
    tgt_domain: target domain
    n_samples: number of samples from target domain include to train dataset
    max_train_samples: number of maximum train samples, default=-1(use all)
    tokenizer: tokenizer
    """
    all_data, slot_exemplars = datareader(tgt_domain, n_samples, data_path)
    # train_data = {"slot_utter": [], "domain": [], "label": [], "template_list": [], "slot_exemplars": {}}
    train_data = {"utter": [], "slot": [], "domain": [], "label": [], "template_list": [], "slot_exemplars": slot_exemplars}
    """
    slot_exemplars_td_only: slot exemplars included only in train domain
    slot_exemplars: total slot exemplars regardless of domain
    """
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain:
            train_data["utter"].extend(dm_data["utter"])
            train_data["slot"].extend(dm_data["slot"])
            train_data["domain"].append(dm_name)
            train_data["label"].extend(dm_data["label"])
            train_data["template_list"].extend(dm_data["template_list"])
            # train_data["slot_exemplars"] = update_dict(train_data["slot_exemplars"], dm_data["slot_exemplars"]) # just add slot exemplars which only in train domain 
            
    val_data = {"utter": [], "slot": [], "domain": [], "label": []}
    test_data = {"utter": [], "slot": [], "domain": [], "label": []}

    val_split = 500 # use 500 data for validation
    train_split = n_samples

    if n_samples == 0:
        # first 500 samples as validation set
        val_data["utter"] = all_data[tgt_domain]["utter"][:val_split]
        val_data["slot"] = all_data[tgt_domain]["slot"][:val_split]
        val_data["domain"]  = [tgt_domain for _ in range(len(val_data["utter"]))]
        val_data["label"] = all_data[tgt_domain]["label"][:val_split]

        # the rest as test set
        test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
        test_data["slot"] = all_data[tgt_domain]["slot"][val_split:]
        test_data["domain"] = [tgt_domain for _ in range(len(test_data["utter"]))]
        test_data["label"] = all_data[tgt_domain]["label"][val_split:]

    else: # delete slot_exemplars_td_only because some of target domain data is added in train data
        # del train_data["slot_exemplars"]
        # train_data["slot_exemplars"] = slot_exemplars

        # first n samples as train set
        train_data["utter"].extend(all_data[tgt_domain]["utter"][:train_split])
        train_data["slot"].extend(all_data[tgt_domain]["utter"][:train_split])
        train_data["domain"].extend([tgt_domain for _ in range(train_split)])
        train_data["label"].extend(all_data[tgt_domain]["label"][:train_split])
        train_data["template_list"].extend(all_data[tgt_domain]["template_list"][:train_split])

        # from n to 500 samples as validation set
        val_data["utter"] = all_data[tgt_domain]["utter"][train_split:val_split]  
        val_data["slot"] = all_data[tgt_domain]["slot"][train_split:val_split]
        val_data["domain"] = [tgt_domain for _ in range(len(val_data["utter"]))]
        val_data["label"] = all_data[tgt_domain]["label"][train_split:val_split]

        # the rest as test set (same as zero-shot)
        test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
        test_data["slot"] = all_data[tgt_domain]["slot"][val_split:]
        test_data["domain"] = [tgt_domain for _ in range(len(test_data["utter"]))]
        test_data["label"] = all_data[tgt_domain]["label"][val_split:]

    dataset_train = Dataset(tokenizer, 
                            train_data["utter"][:max_train_samples], 
                            train_data["slot"][:max_train_samples], 
                            train_data["domain"][:max_train_samples], 
                            train_data["label"][:max_train_samples], 
                            train_data["template_list"][:max_train_samples], 
                            train_data["slot_exemplars"],
                            how_many_augs=num_augs
                            )
                            
    dataset_val = Dataset(tokenizer, 
                            val_data["utter"], 
                            val_data["slot"], 
                            val_data["domain"], 
                            val_data["label"],
                            how_many_augs=num_augs
                            )
    
    dataset_test = Dataset(tokenizer, 
                            test_data["utter"], 
                            test_data["slot"], 
                            test_data["domain"], 
                            test_data["label"],
                            how_many_augs=num_augs
                            )

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_val, dataloader_test