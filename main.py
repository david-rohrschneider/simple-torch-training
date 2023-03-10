from lib.args.args_parser import get_args_parser
from lib.train.trainer import Trainer


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    # Init the trainer
    trainer = Trainer(args=args)
