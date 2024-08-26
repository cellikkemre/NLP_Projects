import pandas as pd

def load_data(file_path, n_rows=None):
    """
    Loads the dataset and returns the specified number of rows.

    :param file_path: The file path of the dataset.
    :param n_rows: Number of rows to load (Default: loads all rows).
    :return: The loaded dataset (DataFrame).
    """
    df = pd.read_csv(file_path, index_col=0)
    if n_rows:
        df = df[:n_rows]
    return df

