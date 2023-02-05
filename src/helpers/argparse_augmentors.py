def parse_augmentors(args, augmentator_mapping):
    augmentor_names = filter(lambda x: x != "", (args.augmentors.split(",")))
    augmentation_params = filter(lambda x: x != "", (args.augmentation_params.split(",")))

    augmentors = []
    for a,p in zip(augmentor_names, augmentation_params):
        augmentor = augmentator_mapping[a]
        augmentor.set_augmentation_percentage(int(p) / 100)
        augmentors.append(augmentor)

    return augmentors