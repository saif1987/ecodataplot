import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame | None:
    """
    Loads data from the specified CSV file path.

    Args:
        csv_path: The path to the input CSV file.

    Returns:
        A pandas DataFrame containing the loaded data, or None if the file is not found.
    """
    try:
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None