import os
import subprocess
import pickle
import logging
import time
import random
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import pickle

y1_set = ["O", "B", "I"]
y2_set = ['O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 'I-geographic_poi', 'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name', 'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type', 'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service', 'I-service', 'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish', 'I-served_dish', 'B-genre',  'I-genre', 'B-current_location', 'I-current_location', 'B-object_select', 'I-object_select', 'B-album', 'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort', 'I-sort', 'B-object_location_type', 'I-object_location_type', 'B-movie_type', 'I-movie_type', 'B-spatial_relation', 'I-spatial_relation', 'B-artist', 'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name', 'I-entity_name', 'B-object_type', 'I-object_type', 'B-playlist_owner', 'I-playlist_owner', 'B-timeRange', 'I-timeRange', 'B-city', 'I-city', 'B-rating_value', 'B-best_rating', 'B-rating_unit', 'B-year', 'B-party_size_number', 'B-condition_description', 'B-condition_temperature']
domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description', 'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name', 'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city', 'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number', 'condition_description', 'condition_temperature']


def init_experiment(params, logger_filename):
    """
    Initialize the experiment:
    - save parameters
    - create a logger
    """
    # save parameters
    get_saved_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_saved_path(params):
    """
    create a directory to store the experiment
    """
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)


def make_syn_data(slot_dict, template_list, num_aug):
    """
    합성 문장을 만드는 함수
    positive augmented data: 1
    negative augmented data: num_aug
    if num_aug == 0: positive augmented data == negative augmented data == 0

    input
    -------
    slot_dict: 슬롯 라벨 및 거기 해당하는 슬롯 예시들의 dictionary
    template_list: 슬롯에 해당하는 단어가 T-slot_label 로 대체된 문장들, 3개 문장의 list로 구성
    how_many: 몇 개 만들지

    output
    -------
    syn_sents: 합성된 문장들
    """
    syn_sents = []
    for i, template in enumerate(template_list):
        if i == 0 and num_aug > 0: 
            how_many = 1
        else: 
            how_many = num_aug

        for _ in range(how_many): # make augmented data replacing slot labels by slot exemplars 
            syn_sent = []
            for word in template.split():
                if "T-" in word:
                    slot_label = word.split('-')[1]
                    slot_exemplar_list = slot_dict[slot_label]
                    slot_exemplar = slot_exemplar_list[random.randint(0, len(slot_exemplar_list) - 1)] # get slot exemplar word
                    syn_sent.append(slot_exemplar)
                else:
                    syn_sent.append(word)

            syn_sent = " ".join(syn_sent)
            syn_sents.append(syn_sent)

    if len(syn_sents) > 0:
        return syn_sents
    else:
        return None


def update_dict(dict1, dict2):
    """
    update contents of dict1 with dict2 which value is list
    """
    assert type(dict1) is dict and type(dict2) is dict
    for key, value in dict2.items():
        if key in dict1.keys():
            dict1[key].extend(value)
        else:
            dict1[key] = value

    return dict1 


def save_params():
    pass


def save_plot(title, xlabel, ylabel, file_path, data_y, data_x = None, dpi=500):
    plt.figure()
    if data_x is not None:
        plt.plot(data_x, data_y)
    else:
        plt.plot(data_y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_path, dpi=dpi)
    plt.close()