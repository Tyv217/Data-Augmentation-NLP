import pytorch_lightning as pl
from argparse import ArgumentParser
from src.models import train_model, hyper_param_search
from src.helpers import set_seed

def main():
    parser = ArgumentParser(conflict_handler = 'resolve')

    # add PROGRAM level args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=str, default="5e-5")
    # parser.add_argument("--deterministic", type=bool, default=True)

    parser.add_argument("--search", type=int, default=0)
    parser.add_argument("--to_search", type=str, default="aug")

    parser.add_argument("--task", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--augmentors", type=str, default="")
    parser.add_argument("--augmentation_params", type=str, default="")
    parser.add_argument("--dataset_percentage", type=int, default=100)
    parser.add_argument("--samples_per_class", type=int)
    parser.add_argument("--invert_saliency", type=int, default = 0)

    parser.add_argument("--logger_dir", type=str, default="")

    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--no_pretrain",  dest='pretrain', action="store_false")
    parser.set_defaults(pretrain=False)
    parser.add_argument("--N_samples", type=int, default=256 * 10)
    parser.add_argument("--N_valid_size", type=int, default=32 * 10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    if args.search == 0:
        train_model(args)
    else:
        hyper_param_search(args)


if __name__ == "__main__":
    main()