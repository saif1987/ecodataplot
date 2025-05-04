import argparse
import os
from data_loader import load_data
from data_processor import reshape_and_clean_data, calculate_gdp_percentages, filter_countries
from plotter import create_bokeh_animation_gif

# --- Configuration ---
OUTPUT_DIR = "plots"
OUTPUT_FILENAME = "asia_trade_gdp_animation.gif"
POPULATION_THRESHOLD = 70_000_000
POPULATION_FILTER_YEAR = 2023 # Use a recent, likely complete year

# List of major Asian countries (adjust as needed based on CSV content and desired scope)
ASIAN_COUNTRIES = [
    'China', 'India', 'Indonesia', 'Pakistan', 'Bangladesh', 'Japan',
    'Philippines', 'Vietnam', 'Iran, Islamic Rep.', 'Turkiye', 'Thailand',
    'Myanmar', 'Korea, Rep.'
]

# Columns required for calculations (ensure these match 'Series' names in CSV)
CALCULATION_COLS = ['GDP (Current US$)', 'Imports (Current US$)', 'Exports (Current US$)']
# Columns required for filtering (ensure this matches 'Series' name in CSV)
FILTER_COLS = ['Population']

def main():
    parser = argparse.ArgumentParser(description="Load, process, and plot economic indicator data.")
    parser.add_argument("csv_file", help="Path to the input CSV file containing economic indicators.")
    args = parser.parse_args()

    # 1. Load Data
    df_raw = load_data(args.csv_file)
    if df_raw is None:
        return # Exit if loading failed

    # 2. Process Data
    df_processed, unique_years = reshape_and_clean_data(df_raw)
    df_processed = calculate_gdp_percentages(df_processed, CALCULATION_COLS)

    # 3. Filter Data
    df_filtered = filter_countries(df_processed, POPULATION_FILTER_YEAR, POPULATION_THRESHOLD, ASIAN_COUNTRIES)

    # 4. Plot Data
    if df_filtered is not None and not df_filtered.empty:
        animation_years = sorted(df_filtered['Year'].unique())
        print(f"Generating animation for years {min(animation_years)} to {max(animation_years)}...")
        create_bokeh_animation_gif(df_filtered, animation_years, OUTPUT_DIR, OUTPUT_FILENAME)
    else:
        print("No data remaining after filtering. Cannot generate plot.")

if __name__ == "__main__":
    main()