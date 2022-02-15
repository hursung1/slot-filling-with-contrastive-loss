from src.utils import (
    y1_set,
    slot_list,
    update_dict
)

import time, random, copy

SLOT_PAD = 0
PAD_INDEX = 0
UNK_INDEX = 1

def read_file(filepath, domain, num_neg_tems=2):
    """
    파일을 읽어서 dictionary에 저장

    returns
    ----------
    - data_dict: 각 문장과 그에 대한 slot label sequence, 그리고 template sentence를 dictionary 형태로 가짐.
    """
    seed = time.time()
    utter_list, y1_list, y2_list, slot_desc_list, template_list = [], [], [], [], []
    """
    utter_list: list of utterances
    y1_list: BIO tagging only, shape is same as utter_list
    y2_list: BIO tagging with slot label, shape is same as utter_list
    template_list: list of templates, list of list, and inner list consists of three templates(one positive, two negatives), [[pos_tem,neg_tem1,neg_tem2], [pos_tem,neg_tem1,neg_tem2], ...]
    """
    slot_exemplar_dict = {}
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()  # text \t label
            splits = line.split("\t")
            utter = splits[0]
            tokens = splits[0].split()
            l2_list = splits[1].split()

            # utter_list.append(utter)
            # y2_list.append(l2_list)

            slot_word = []
            slot_label = None 
            labels_per_utter = []
            BIO_with_slot_dict = {}
            for i, (word, l) in enumerate(zip(tokens, l2_list)):
                if "B" in l:
                    tag, slot_label = l.split('-')
                    BIO_with_slot_dict[slot_label] = ["O" for _ in range(len(l2_list))]
                    BIO_with_slot_dict[slot_label][i] = tag # "B"
                    slot_word.append(word)
                
                elif "I" in l:
                    tag, slot_label = l.split('-')
                    BIO_with_slot_dict[slot_label][i] = tag # "I"
                    slot_word.append(word)
                
                else:
                    # l1_list.append(y1_set.index(l))
                    if len(slot_word) != 0: # when slot words exist
                        slot_word = " ".join(slot_word)
                        try:
                            if slot_word not in slot_exemplar_dict[slot_label]:
                                slot_exemplar_dict[slot_label].append(slot_word)
                        except KeyError:
                            if slot_label is not None:
                                slot_exemplar_dict[slot_label] = [slot_word]
                            else:
                                print("slot_label should not be None")
                                exit()
                                
                        # utter_list.append(utter)
                        # y1_list.append(l1_list)
                        # y2_list.append(l2_list)
                        # slot_desc_list.append(slot_label)

                        slot_word = []
                        slot_label = None
            
            # slot_list = list(BIO_with_slot_dict.keys())
            # l2_list = list(BIO_with_slot_dict.values())
            
            for key, value in BIO_with_slot_dict.items():
                utter_list.append(utter)
                slot_desc_list.append(key)
                y1_list.append(value)

                """
                template_each_sample[0] is correct template
                from template_each_sample[1] to template_each_sample[n] are incorrect template (replace correct slots with other slots)
                """
                template_each_sample = [[] for _ in range(num_neg_tems + 1)] # pos_tem 1 & neg_tems
                for token_idx, tag in enumerate(value):
                    if tag == "B":
                        template_each_sample[0].append("T-"+key) # positive template
                        _slot_list = copy.deepcopy(slot_list)
                        random.shuffle(_slot_list)
                        idx = 0
                        for neg_tem_idx in range(1, num_neg_tems + 1): # negative template
                            if _slot_list[idx] == key:
                                idx += 1
                                
                            template_each_sample[neg_tem_idx].append("T-" + _slot_list[idx])
                            idx += 1

                    elif tag == "I":
                        continue
                    else: # tag == "O"
                        for tem_idx in range(num_neg_tems + 1):
                            template_each_sample[tem_idx].append(tokens[token_idx])
                
                for i in range(num_neg_tems + 1):
                    template_each_sample[i] = " ".join(template_each_sample[i])

                template_list.append(template_each_sample)
    
    # random shuffle
    random.Random(seed).shuffle(utter_list)
    random.Random(seed).shuffle(slot_desc_list)
    random.Random(seed).shuffle(y1_list)
    random.Random(seed).shuffle(template_list)

    data_dict = {"utter": utter_list, "slot": slot_desc_list, "label": y1_list, "template_list": template_list, "slot_exemplars": slot_exemplar_dict}

    return data_dict


def binarize_data(data):
    """
    - y1을 "B", "I", "O"에서 1, 2, 0으로 바꿈
    - x dictionary 형태로 저장된 string data를 bert tokenizer를 통해 binarize하는 함수.
    - x bert tokenizer는 한 단어에 대해 subword token화 함 --> y1과 y2가 subword화 된 단어들에 대해 똑같이 늘려져야 함.

    Argument
    ----------
    - data: utter/y1/y2/template_list를 string 형태로 갖는 dictionary 
    - x tokenizer: bert tokenizer

    Returns
    ----------
    - data_bin: bert tokenizer에 {$data} 를 통과시킨 후의 output 
    """
    data_bin = {"utter": data['utter'], "slot": data["slot"], "label": [], "template_list": data['template_list'], "slot_exemplars": data['slot_exemplars']}
    
    for tag_list in data['label']:
        label_bin = []
        for tag in tag_list:
            label_bin.append(y1_set.index(tag))

        data_bin['label'].append(label_bin)

    return data_bin
    

# def datareader(tokenizer, aug_prob):
def datareader(tgt_domain, n_samples, data_path):
    # logger.info("Loading and processing data ...")

    data = {"AddToPlaylist": {}, "BookRestaurant": {}, "GetWeather": {}, "PlayMusic": {}, "RateBook": {}, "SearchCreativeWork": {}, "SearchScreeningEvent": {}}
    slot_exemplar_dict_with_domain = {"AddToPlaylist": {}, "BookRestaurant": {}, "GetWeather": {}, "PlayMusic": {}, "RateBook": {}, "SearchCreativeWork": {}, "SearchScreeningEvent": {}}
    slot_exemplar_dict = {}

    if n_samples == 0:
        tgt_domain = None
        
    # load data
    for key, _ in data.items():
        data_each = read_file(f"{data_path}/{key}/{key}.txt", key)
        data[key] = data_each
        slot_exems = data[key]["slot_exemplars"]
        slot_exemplar_dict_with_domain[key] = slot_exems
        slot_exemplar_dict = update_dict(slot_exemplar_dict, slot_exems)

    for key, value in data.items():
        data[key] = binarize_data(value)

    return data, slot_exemplar_dict