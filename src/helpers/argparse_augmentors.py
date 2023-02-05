from ..models import Synonym_Replacer, Back_Translator, Insertor, Deletor

def parse_augmentors(args):
    augmentor_names = (args.augmentors.split(",")).filter(lambda x: x != "")
    augmentation_params = (args.augmentation_params.split(",")).filter(lambda x: x != "")

    augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en", "de"), "in": Insertor("english"), "de": Deletor()}

    augmentors = []
    for a,p in zip(augmentor_names, augmentation_params):
        augmentor = augmentator_mapping[a]
        augmentor.set_augmentation_percentage(int(p))
        augmentors.append(augmentor)

    return augmentors