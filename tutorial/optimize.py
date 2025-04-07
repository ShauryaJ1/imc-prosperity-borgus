import itertools
import subprocess
import re
import json
import os
from main import Trader


class Optimize:
    @staticmethod
    def grid_search(trader, param_grid: dict, params_filename="trader_params.json") -> tuple:
        """
        Performs a grid search over the given hyperparameters to maximize total profit.

        Parameters:
            trader (Trader): An instance of the Trader class.
            param_grid (dict): Dictionary where keys are trader attribute names (or nested ones using dot notation,
                               e.g., "idealProfits.KELP") and values are iterables of possible values.
            params_filename (str): Filename to write the updated parameters to.

        Returns:
            A tuple (best_params, best_profit) where best_params is a dictionary of the optimal hyperparameters
            and best_profit is the corresponding total profit.
        """

        best_configs = dict()

        # Get all parameter names and the corresponding list of possible values.
        keys = list(param_grid.keys())
        values_list = list(param_grid.values())

        # Iterate over every combination of parameters.
        for combination in itertools.product(*values_list):
            params = dict(zip(keys, combination))

            # Create a dictionary to hold the parameters in a structured way.
            trader_params = {}

            # Update trader instance and also build the dictionary.
            for param, value in params.items():
                if "." in param:
                    parts = param.split(".")
                    # Ensure the top-level key exists.
                    if parts[0] not in trader_params:
                        trader_params[parts[0]] = {}
                    trader_params[parts[0]][parts[1]] = value

                    # Also update the trader instance.
                    attr = getattr(trader, parts[0], None)
                    if attr is None or not isinstance(attr, dict):
                        raise ValueError(f"Trader does not have a dict attribute named {parts[0]}")
                    attr[parts[1]] = value
                else:
                    trader_params[param] = value
                    setattr(trader, param, value)

            # Write the current parameters to a file.
            with open(params_filename, "w") as f:
                json.dump(trader_params, f)

            env = os.environ.copy()
            env["PATH"] = env["PATH"] + r";C:\Users\samar\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts"

            # Run the backtest command and pass the parameters file.
            # main.py should be modified to read parameters from the file specified by the '--params' flag.
            result = subprocess.run(
                ["prosperity3bt", "main_samarth.py", "0", "--no-out"],
                capture_output=True, text=True
            )
            output = result.stdout
            # print(output)

            # Parse the profit from the output.
            profit = Optimize.parse_profit(output)

            # Log the current test and its profit.
            print(f"Tested params: {params}, Profit: {profit}")

            # Update the best parameters if this run is better.
            if profit is not None:
                best_configs[profit] = params.copy()
                if len(best_configs) > 10:
                    least_key = min(best_configs.keys())
                    del best_configs[least_key]

        # Optionally remove the parameters file.
        if os.path.exists(params_filename):
            os.remove(params_filename)

        print(best_configs)

        return best_configs

    @staticmethod
    def parse_profit(output: str) -> int:
        """
        Parses the output text from the backtesting command to extract the total profit.
        Expected format in the output is a line starting with "Total profit:".

        Parameters:
            output (str): The output text from the backtest command.

        Returns:
            The extracted profit as an integer, or None if it cannot be parsed.
        """
        match = re.search(r"Total profit:\s*([\d,]+)", output)
        if match:
            profit_str = match.group(1).replace(",", "")
            try:
                return int(profit_str)
            except ValueError:
                return None
        return None

if __name__ == "__main__":
    Optimize.grid_search(Trader(), {

    })

