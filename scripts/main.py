import argparse
import os
from data_loader import load_data
from data_processor import reshape_and_clean_data, calculate_gdp_percentages, calculate_gdp_per_capita, filter_countries
from plotter import create_seaborn_animation, create_gdp_per_capita_animation

# --- Configuration ---
OUTPUT_DIR = "plots"
OUTPUT_FILENAME = "asia_trade_gdp_animation.gif" # Revert extension to .gif
POPULATION_THRESHOLD = 70_000_000
POPULATION_FILTER_YEAR = 2023 # Use a recent, likely complete year

# Plot types
PLOT_TYPE_TRADE = "trade_vs_gdp"
PLOT_TYPE_GDP_PER_CAPITA = "gdp_per_capita_vs_year"
ALL_PLOT_TYPES = [PLOT_TYPE_TRADE, PLOT_TYPE_GDP_PER_CAPITA]
# List of major Asian countries (adjust as needed based on CSV content and desired scope)
ASIAN_COUNTRIES = [
    # 'China', 
    'India', 
    # 'Indonesia', 
    'Pakistan', 
    'Bangladesh', 
    'Philippines', 
    'Viet Nam',
    # 'Thailand',
    # 'Myanmar'
]

# Columns required for calculations (ensure these match 'Series' names in CSV)
CALCULATION_COLS = ['GDP (current US$)', 'Imports of goods and services (current US$)', 'Exports of goods and services (current US$)']
# Columns required for filtering and other calculations (ensure these match 'Series' names in CSV)
REQUIRED_SERIES = ['Population, total'] # 'GDP (current US$)' is already in CALCULATION_COLS

def prepare_trade_plot_data(df):
    """Prepares data specifically for the trade vs GDP plot, including interpolation."""
    plot_cols = ['Country', 'Year', 'Imports (% GDP)', 'Exports (% GDP)']
    if not all(col in df.columns for col in plot_cols):
        print("Warning: Missing required columns for trade plot data preparation.")
        return pd.DataFrame()

    df_trade = df[plot_cols].copy()
    print("Interpolating missing import/export percentages for trade plot...")
    for col in ['Imports (% GDP)', 'Exports (% GDP)']:
        df_trade[col] = df_trade.groupby('Country')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both', axis=0)
        )
    df_trade = df_trade.dropna(subset=['Imports (% GDP)', 'Exports (% GDP)'])
    print(f"Trade plot data prepared with {len(df_trade)} rows.")
    return df_trade

def prepare_gdp_per_capita_plot_data(df):
    """Prepares data specifically for the GDP per capita plot, including interpolation."""
    plot_cols = ['Country', 'Year', 'GDP per capita']
    if not all(col in df.columns for col in plot_cols):
        print("Warning: Missing required columns for GDP per capita plot data preparation.")
        return pd.DataFrame()

    df_gdp_pc = df[plot_cols].copy()
    print("Interpolating missing GDP per capita data for plot...")
    df_gdp_pc['GDP per capita'] = df_gdp_pc.groupby('Country')['GDP per capita'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both', axis=0)
    )
    df_gdp_pc = df_gdp_pc.dropna(subset=['GDP per capita'])
    print(f"GDP per capita plot data prepared with {len(df_gdp_pc)} rows.")
    return df_gdp_pc

def main():
    parser = argparse.ArgumentParser(description="Load, process, and plot economic indicator data.")
    parser.add_argument(
            "csv_file",
            nargs='?',  # Makes the argument optional
            default='./data/economic_indicators_1975-2025_20250504.csv', # Sets the default value
            help="Path to the input CSV file containing economic indicators. Defaults to './data/economic_indicators_1975-2025_20250504.csv' if not provided."
        )
   
    parser.add_argument(
            "--plot-types",
            nargs='*',
            choices=ALL_PLOT_TYPES + ["all"],
            default=["all"],
            help=f"Specify which plot(s) to generate. Choices: {', '.join(ALL_PLOT_TYPES)}, or 'all' for all plots."
        )
    args = parser.parse_args()
    csv_file = args.csv_file

    selected_plot_types = args.plot_types
    if "all" in selected_plot_types:
        plot_types_to_generate = ALL_PLOT_TYPES
    else:
        plot_types_to_generate = [ptype for ptype in selected_plot_types if ptype in ALL_PLOT_TYPES]

    if not plot_types_to_generate:
        print("No valid plot types selected. Exiting.")
        return

    # 1. Load Data
    df_raw = load_data(csv_file)
    if df_raw is None:
        return # Exit if loading failed

    # 2. Process Data (reshape, clean, calculate base indicators)
    df_processed, unique_years = reshape_and_clean_data(df_raw)
    if df_processed is None: return

    df_processed = calculate_gdp_percentages(df_processed, CALCULATION_COLS)
    if df_processed is None: return

    df_processed = calculate_gdp_per_capita(df_processed) # New calculation
    if df_processed is None: return

    # 3. Filter Data
    df_filtered_countries = filter_countries(df_processed, POPULATION_FILTER_YEAR, POPULATION_THRESHOLD, ASIAN_COUNTRIES)
    if df_filtered_countries is None or df_filtered_countries.empty:
        print("No data remaining after filtering countries. Cannot generate plots.")
        return

    # 4. Plot Data
    if PLOT_TYPE_TRADE in plot_types_to_generate:
        print(f"\n--- Generating {PLOT_TYPE_TRADE} plot ---")
        df_trade_plot_ready = prepare_trade_plot_data(df_filtered_countries.copy())
        if not df_trade_plot_ready.empty:
            animation_years = sorted(df_trade_plot_ready['Year'].unique())
            print(f"Generating trade animation for years {min(animation_years)} to {max(animation_years)}...")
            create_seaborn_animation(
                df_trade_plot_ready,
                animation_years,
                OUTPUT_DIR,
                "asia_trade_gdp_animation.gif"
            )
        else:
            print(f"No data for {PLOT_TYPE_TRADE} plot after preparation.")

    if PLOT_TYPE_GDP_PER_CAPITA in plot_types_to_generate:
        print(f"\n--- Generating {PLOT_TYPE_GDP_PER_CAPITA} plot ---")
        df_gdp_pc_plot_ready = prepare_gdp_per_capita_plot_data(df_filtered_countries.copy())
        if not df_gdp_pc_plot_ready.empty:
            animation_years = sorted(df_gdp_pc_plot_ready['Year'].unique())
            print(f"Generating GDP per capita animation for years {min(animation_years)} to {max(animation_years)}...")
            create_gdp_per_capita_animation(
                df_gdp_pc_plot_ready,
                animation_years,
                OUTPUT_DIR,
                "asia_gdp_per_capita_animation.gif"
            )
        else:
            print(f"No data for {PLOT_TYPE_GDP_PER_CAPITA} plot after preparation.")

if __name__ == "__main__":
    main()