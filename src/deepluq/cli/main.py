from argparse import ArgumentParser


def run():
    parser = ArgumentParser()
    parser.add_argument('--model', default=None, help='model name')
    args, unknown_args = parser.parse_known_args()
    print(args.model)
