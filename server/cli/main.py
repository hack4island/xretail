import argparse, sys
from inference import add_inference_subparser
from server import add_server_subparser

def get_arg_parsers():
    verbosity_parser = argparse.ArgumentParser(add_help=False)
    verbosity_parser.add_argument(
        "-v", "--verbosity", action="count", default=0, help="Increase output verbosity"
    )

    arg_parser = argparse.ArgumentParser(
        description="Xretail CLI",
        parents=[verbosity_parser],
    )

    command_subparsers = arg_parser.add_subparsers(
        title="available commands", metavar="command [options ...]"
    )

    add_inference_subparser(command_subparsers, [verbosity_parser])
    add_server_subparser(command_subparsers, [verbosity_parser])

    return arg_parser

if __name__ == "__main__":
    arg_parser = get_arg_parsers()
    args = arg_parser.parse_args()
    verbosity = args.verbosity

    if hasattr(args, "func"):
        args.func(args)
    else:
        arg_parser.print_help()
        sys.exit(1)
