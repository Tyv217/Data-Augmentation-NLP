def parse_augmentors(args, augmentator_mapping):
    augmentor_names = (args.augmentors.split(",")).filter(lambda x: x != "")
    augmentation_params = (args.augmentation_params.split(",")).filter(lambda x: x != "")

    augmentors = []
    for a,p in zip(augmentor_names, augmentation_params):
        augmentor = augmentator_mapping[a]
        augmentor.set_augmentation_percentage(int(p) / 100)
        augmentors.append(augmentor)

    return augmentors