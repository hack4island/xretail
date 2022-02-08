from pathlib import Path
from price_reader import PriceTag, PriceReader

def add_inference_subparser(subparsers, parent_parsers):
    parsing_argparser = subparsers.add_parser(
        "inference",
        help="Run an inference on a given input file and return product informations",
        parents=parent_parsers,
    )
    parsing_argparser.add_argument(
        "--input", type=str, help="Path to the input file"
    )
    parsing_argparser.set_defaults(func=_run_inference)
    return parsing_argparser


def _run_inference(namespace):
    run_inference(
        input_path=Path(namespace.input),
    )

def run_inference(input_path):
    price_tag = PriceTag(input_path)
    price_reader = PriceReader()

    # Loading price reader
    price_reader.load()

    # Extract product from price tag
    product = price_reader.get_product(price_tag)

    print(product)
