def parse_augmentors(args, augmentator_mapping):
    augmentor_names = filter(lambda x: x != "", (args.augmentors.split(",")))
    augmentation_params = filter(lambda x: x != "", (args.augmentation_params.split(",")))

    augmentors_on_words = []
    augmentors_on_tokens = []
    for a,p in zip(augmentor_names, augmentation_params):
        augmentor = augmentator_mapping[a]
        augmentor.set_augmentation_percentage(int(p) / 100)
        if augmentor.operate_on_tokens:
            augmentors_on_tokens.append(augmentor)
        else:
            augmentors_on_words.append(augmentor)

    return augmentors_on_words, augmentors_on_tokens