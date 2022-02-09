from app import app

def add_server_subparser(subparsers, parent_parsers):
    parsing_argparser = subparsers.add_parser(
        "server",
        help="Run an inference server",
        parents=parent_parsers,
    )
    parsing_argparser.set_defaults(func=run_server)

    return parsing_argparser


def run_server(namespace):
    app.run(host="0.0.0.0", port="8000")

