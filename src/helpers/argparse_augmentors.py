from ..models.data_augmentors import Synonym_Replacer, Back_Translator, Insertor, Deletor, CutOut, CutMix, MixUp
from copy import deepcopy
AUGMENTATOR_MAPPING = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix(), "mu": MixUp()}

def parse_augmentors(args):
    if args.use_default_augmentation_params != 0:
        return parse_augmentors_string(args.augmentors, args.augmentation_params)
    augmentor_names = args.augmentors

    if augmentor_names == '':
        return [], []
    
    if args.task == "classify" or args.task == "classify_saliency":
        task = args.dataset
    else:
        task = args.task

    augmentation_param_mapping = {
        "ag_news": { 
            "sr": [30],
            "in": [15],
            "de": [25],
            "sr,in,de": [40,10,5],
            "bt": [500],
            "co": [90],
            "mu": [90],
            "cm": [65],
        },
        "babe": {
            "sr": [40],
            "in": [20],
            "de": [35],
            "sr,in,de": [5,40,30],
            "bt": [150],
            "co": [25],
            "mu": [25],
            "cm": [30],
        },
        "cola": {
            "sr": [0],
            "in": [45],
            "de": [45],
            "sr,in,de": [0,0,0],
            "bt": [200],
            "co": [30],
            "mu": [95],
            "cm": [95],
        },
        "sst2": {
            "sr": [0],
            "in": [45],
            "de": [45],
            "sr,in,de": [0,0,0],
            "bt": [200],
            "co": [30],
            "mu": [94],
            "cm": [95],
        },
        "qnli": {
            "sr": [0],
            "in": [45],
            "de": [45],
            "sr,in,de": [0,0,0],
            "bt": [200],
            "co": [30],
            "mu": [94],
            "cm": [95],
        },
        "trec": {
            "sr": [0],
            "in": [0],
            "de": [0],
            "sr,in,de": [0,0,0],
            "bt": [0],
            "co": [0],
            "mu": [0],
            "cm": [0],
        },
        "translate": {
            "sr": [10],
            "in": [10],
            "de": [5],
            "sr,in,de": [10,10,5],
            "bt": [50]
        },
        "language_model": {
            "sr": [10],
            "in": [10],
            "de": [5],
            "sr,in,de": [10,10,5],
            "bt": [50]
        },
    }

    augmentors_on_words = []
    augmentors_on_tokens = []
    
    augmentation_params = augmentation_param_mapping[task][augmentor_names]

    for p, n in zip(augmentation_params, list(filter(lambda x: x != "", (args.augmentors.split(","))))):
        augmentor = AUGMENTATOR_MAPPING[n]
        augmentor.set_augmentation_percentage(p / 100)
        if augmentor.operate_on_embeddings:
            augmentors_on_tokens.append(augmentor)
        else:
            augmentors_on_words.append(augmentor)

    return augmentors_on_words, augmentors_on_tokens

def parse_policy(file_name):
    policy = []
    with open(file_name, 'r') as f:
        for line in f:
            augmentors = line.split(";")
            subpolicy = []
            for augmentor_details in augmentors:
                aug_name = augmentor_details.split(",")[0]
                aug_prob = augmentor_details.split(",")[1]
                augmentor = deepcopy(AUGMENTATOR_MAPPING[aug_name])
                augmentor.augmentation_percentage = float(aug_prob)
                subpolicy.append(augmentor)
            policy.append(subpolicy)
    return policy

def parse_augmentors_string(augmentor_names, augmentation_params):
    augmentor_names = filter(lambda x: x != "", (augmentor_names.split(",")))
    augmentation_params = filter(lambda x: x != "", (augmentation_params.split(",")))

    augmentors_on_words = []
    augmentors_on_tokens = []
    for a,p in zip(augmentor_names, augmentation_params):
        augmentor = AUGMENTATOR_MAPPING[a]
        augmentor.set_augmentation_percentage(int(p) / 100)
        if augmentor.operate_on_embeddings:
            augmentors_on_tokens.append(augmentor)
        else:
            augmentors_on_words.append(augmentor)

    return augmentors_on_words, augmentors_on_tokens

def parse_augmentors_int(augmentor_names, augmentation_params):
    augmentors_on_words = []
    augmentors_on_tokens = []
    for a,p in zip(augmentor_names, augmentation_params):
        augmentor = AUGMENTATOR_MAPPING[a]
        augmentor.set_augmentation_percentage(int(p) / 100)
        if augmentor.operate_on_embeddings:
            augmentors_on_tokens.append(augmentor)
        else:
            augmentors_on_words.append(augmentor)

    return augmentors_on_words, augmentors_on_tokens