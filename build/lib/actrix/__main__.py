import argparse
from actrix.cli.models import add_models_commands

def main():
    parser = argparse.ArgumentParser(prog="actrix")
    subparsers = parser.add_subparsers(dest="command")
    
    # Models subcommands
    models_parser = subparsers.add_parser("models", help="Model operations")
    models_sub = models_parser.add_subparsers(dest="subcommand")
    add_models_commands(models_sub)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()