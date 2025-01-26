from tabulate import tabulate
from ..models.registry import list_models, get_model_params

def add_models_commands(subparsers):
    # List models
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.set_defaults(handler=lambda _: print("\n".join(list_models())))

    # Model parameters
    params_parser = subparsers.add_parser("params", help="Show model parameters")
    params_parser.add_argument("name", help="Model name to show parameters for")
    params_parser.set_defaults(handler=lambda args: show_params(args.name))

def show_params(name: str):
    """
    Displays the parameters and their metadata for a given model configuration in a tabular format.

    Args:
        name (str): Name of the model configuration.
    """
    try:
        params = get_model_params(name)
        print(f"Parameters for {name}:")
        # Prepare data for tabulate
        table_data = [(k, v["type"], v["help"]) for k, v in params.items()]
        print(tabulate(table_data, headers=["Parameter", "Type", "Description"], tablefmt="pretty"))
    except ValueError as e:
        print(f"Error: {str(e)}")

        