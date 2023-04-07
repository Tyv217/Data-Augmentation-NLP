from ..models.data_augmentors import Synonym_Replacer, Back_Translator, Insertor, Deletor, CutOut, CutMix, MixUp
AUGMENTATOR_MAPPING = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix(), "mu": MixUp()}

def parse_augmentors(args):
    augmentor_names = args.augmentors

    if augmentor_names == '':
        return [], []
    
    if args.task == "classify":
        task = args.dataset
    else:
        task = args.task

    augmentation_param_mapping = {
        "ag_news": {
            "sr": [0],
            "in": [0],
            "de": [24],
            "sr,in,de": [0,0,0],
            "bt": [0],
            "co": [0],
            "mu": [0],
            "cm": [0],
        },
        "babe": {
            "sr": [40],
            "in": [20],
            "de": [35],
            "sr,in,de": [5,40,30],
            "bt": [0],
            "co": [25],
            "mu": [25],
            "cm": [30],
        },
        "glue": {
            "sr": [0],
            "in": [0],
            "de": [0],
            "sr,in,de": [0,0,0],
            "bt": [0],
            "co": [0],
            "mu": [0],
            "cm": [0],
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
            "sr": [0],
            "in": [0],
            "de": [0],
            "sr,in,de": [0,0,0],
            "bt": [0],
            "co": [0],
            "mu": [0],
            "cm": [0],
        },
        "language_model": {
            "sr": [0],
            "in": [0],
            "de": [0],
            "sr,in,de": [0,0,0],
            "bt": [0],
            "co": [0],
            "mu": [0],
            "cm": [0],
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