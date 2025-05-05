import pandas as pd
import numpy as np

def reshape_and_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[int] | None]:
    """
    Reshapes the DataFrame from wide to long format, cleans year column,
    and pivots indicators into columns.

    Args:
        df: The raw DataFrame loaded from the CSV.

    Returns:
        A tuple containing the processed DataFrame and a sorted list of unique years,
        or (None, None) if processing fails.
    """
    if df is None:
        return None, None

    print("Reshaping and cleaning data...")
    # Identify year columns (assuming format YR####)
    year_columns = [col for col in df.columns if col.startswith('YR')]
    id_vars = ['economy', 'series', 'Country', 'Series'] # Ensure these columns exist

    if not all(col in df.columns for col in id_vars):
        print(f"Error: Input DataFrame missing one or more required ID columns: {id_vars}")
        return None, None

    # Melt the DataFrame to long format
    df_long = pd.melt(df, id_vars=id_vars, value_vars=year_columns,
                      var_name='Year', value_name='Value')

    # Clean Year column
    df_long['Year'] = df_long['Year'].str.replace('YR', '').astype(int)

    # Pivot the table to have indicators as columns
    try:
        df_pivot = df_long.pivot_table(index=['Country', 'Year'],
                                       columns='Series', values='Value').reset_index()
        print("Data reshaped successfully.")
        return df_pivot, sorted(df_pivot['Year'].unique())
    except KeyError:
         print("Error: Could not find 'Country', 'Year', 'Series', 'Value' columns after melting.")
         print("Please check the CSV structure and column names used during melting.")
         return None, None
    except Exception as e:
        print(f"An error occurred during pivoting: {e}")
        return None, None

def calculate_gdp_percentages(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame | None:
    """
    Calculates Imports and Exports as a percentage of GDP.

    Args:
        df: The pivoted DataFrame with indicators as columns.
        required_cols: List of column names required for calculation (GDP, Imports, Exports).

    Returns:
        The DataFrame with added percentage columns, or None if required columns are missing.
    """
    if df is None: return None
    print("Calculating GDP percentages...")
    if not all(col in df.columns for col in required_cols):
        print("Error: Missing required indicator columns for percentage calculation.")
        print(f"Required: {required_cols}")
        print(f"Found: {df.columns.tolist()}")
        return None

    # Calculate percentages, handle potential division by zero or NaN
    df['Imports (% GDP)'] = (df['Imports of goods and services (current US$)'] / df['GDP (current US$)']) * 100
    df['Exports (% GDP)'] = (df['Exports of goods and services (current US$)'] / df['GDP (current US$)']) * 100

    # Replace potential infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("GDP percentages calculated.")
    return df

def filter_countries(df: pd.DataFrame, year_to_filter: int, pop_threshold: int, countries_list: list[str]) -> pd.DataFrame | None:
    """Filters the DataFrame for specified countries based on population."""
    if df is None: return None
    print(f"Filtering for countries in list, with population > {pop_threshold/1e6:.0f}M in {year_to_filter}...")

    pop_col = 'Population, total' # Ensure this column name matches your data after pivoting
    if pop_col not in df.columns:
        print(f"Error: Population column '{pop_col}' not found for filtering.")
        return None

    pop_data = df[df['Year'] == year_to_filter]
    large_countries = pop_data[pop_data[pop_col] > pop_threshold]['Country'].unique()
    filtered_countries = [country for country in large_countries if country in countries_list]

    print(f"Selected Countries: {filtered_countries if filtered_countries else 'None'}")

    if not filtered_countries:
        return pd.DataFrame() # Return empty DataFrame if no countries selected

    filtered_df = df[df['Country'].isin(filtered_countries)].copy()

    # Interpolate missing import/export percentages within each country group
    print("Interpolating missing import/export percentages...")
    plot_cols = ['Country', 'Year', 'Imports (% GDP)', 'Exports (% GDP)']
    # Use transform for potentially more robust interpolation within groups
    for col in plot_cols[2:]: # Iterate through 'Imports (% GDP)' and 'Exports (% GDP)'
        filtered_df[col] = filtered_df.groupby('Country')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both', axis=0)
        )

    filtered_df = filtered_df[plot_cols].dropna(subset=['Imports (% GDP)', 'Exports (% GDP)'])
    print(f"Filtered data contains {len(filtered_df)} rows.")
    return filtered_df