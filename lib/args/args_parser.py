"""Module returning the ArgumentParser"""
from argparse import ArgumentParser


def get_args_parser(add_help=True) -> ArgumentParser:
    """Parse all args from user input"""

    parser = ArgumentParser(
        description="Simple PyTorch Training 🧠",
        add_help=add_help)

    parser.add_argument(
        "--model",
        type=str,
        help="model name",
        required=True
    )
    parser.add_argument(
        "--data-path",
        default=".",
        type=str,
        help="Path to your dataset - default: %(default)s"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Your device - cuda/cpu - default: %(default)s"
    )
    parser.add_argument(
        "--batch-size", "-b",
        default=32,
        type=int,
        help="#Images per GPU, the total batch size is #GPUs x batch_size - default: %(default)s",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="N",
        help="Number of total epochs to run - default: %(default)s"
    )
    parser.add_argument(
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="Number of data loading workers - default: %(default)s"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        type=str,
        help="Path to save outputs - default: %(default)s"
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        type=str,
        help="Path of checkpoint file - No default"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="Start epoch - default: %(default)s"
    )
    parser.add_argument(
        "--cache-dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--eval",
        help="Only evaluate the models performance",
        action="store_true",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training"
    )
    parser.add_argument(
        "--weights-enum",
        default=None,
        type=str,
        help="Torchvision enum name for getting weights and transforms"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Method to perform - classification/detection/segmentation",
        required=True
    )
    parser.add_argument(
        "--hyp-config",
        default=None,
        type=str,
        help="Path to your config file for hyperparams - default: %(default)s"
    )

    return parser
