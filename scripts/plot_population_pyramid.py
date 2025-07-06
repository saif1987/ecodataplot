import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from matplotlib.patches import Patch # Import Patch for custom legend handles


def plot_population_pyramid(csv_path: str, output_dir: str, output_filename: str, plot_type: str = 'percent'):
    """
    Generates and saves a population pyramid plot.

    Args:
        csv_path: Path to the CSV file containing population data.
                  Expected columns: 'GROUP' (age group), 'Male Population', 'Female Population'.
        output_dir: Directory to save the output plot.
        output_filename: Name for the output plot file (e.g., .png).
        plot_type: 'percent' to plot percentages of total population,
                   'population' to plot raw numbers,
                   or 'acc_pct' to plot accumulated percentage of population above each age.
                   Defaults to 'percent'.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        return

    required_cols = ['GROUP', 'Male Population', 'Female Population']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return

    # Ensure 'GROUP' column is treated as string/category for plotting labels
    df['GROUP'] = df['GROUP'].astype(str)

    # Filter out rows where 'GROUP' is an empty string or the string 'nan',
    # as these are often summary rows or invalid age group entries.
    df = df[df['GROUP'].str.strip() != '']
    df = df[df['GROUP'].str.strip().str.lower() != 'nan'] 

    # Clean and ensure population columns are numeric
    # Remove commas from the string representation of numbers before converting to numeric
    df['Male Population'] = df['Male Population'].astype(str).str.replace(',', '', regex=False)
    df['Male Population'] = pd.to_numeric(df['Male Population'], errors='coerce').fillna(0)
    
    df['Female Population'] = df['Female Population'].astype(str).str.replace(',', '', regex=False)
    df['Female Population'] = pd.to_numeric(df['Female Population'], errors='coerce').fillna(0)
    
    print("\n--- Debug: DataFrame Summary for Population Pyramid ---")
    print(f"DataFrame head:\n{df.head()}")
    print(f"\nDataFrame info:\n")
    df.info()
    print(f"\nDataFrame describe:\n{df.describe()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print("----------------------------------------------------")

    # Create a numerical sort key for the 'GROUP' column
    # This handles '100+' by assigning a large number for correct sorting
    df['sort_key'] = df['GROUP'].apply(lambda x: int(x) if x.isdigit() else (100 if x == '100+' else np.inf))
    
    # Sort age groups numerically using the new sort_key
    df = df.sort_values(by='sort_key', ascending=True)

    print(f"\n--- Debug: Unique GROUP values after filtering and numerical sorting: {df['GROUP'].unique()} ---")
    print("----------------------------------------------------")

    # Calculate percentages for plotting relative to the total population
    total_male = df['Male Population'].sum()
    total_female = df['Female Population'].sum()
    total_population = total_male + total_female

    # Handle case where total population is zero to avoid division by zero
    if total_population == 0:
        df['Male_Percent'] = 0
        df['Female_Percent'] = 0
    else:
        df['Male_Percent'] = (df['Male Population'] / total_population) * 100
        df['Female_Percent'] = (df['Female Population'] / total_population) * 100

    # Calculate accumulated percentages in reverse (percentage of population at or above this age)
    # Ensure the DataFrame is sorted correctly before this step
    df['Male_Acc_Percent_Rev'] = df['Male_Percent'].iloc[::-1].cumsum().iloc[::-1]
    df['Female_Acc_Percent_Rev'] = df['Female_Percent'].iloc[::-1].cumsum().iloc[::-1]
    
    # Initialize data series for plotting
    male_common_data = pd.Series([])
    male_excess_data = pd.Series([])
    female_common_data = pd.Series([])
    female_excess_data = pd.Series([])

    # Determine which columns to plot and what the x-axis label should be
    if plot_type == 'percent':
        # Calculate common and excess percentages
        df['Common_Percent'] = df[['Male_Percent', 'Female_Percent']].min(axis=1)
        df['Excess_Male_Percent'] = df['Male_Percent'] - df['Common_Percent']
        df['Excess_Female_Percent'] = df['Female_Percent'] - df['Common_Percent']

        male_common_data = df['Common_Percent']
        male_excess_data = df['Excess_Male_Percent']
        female_common_data = df['Common_Percent']
        female_excess_data = df['Excess_Female_Percent']

        x_label = "Percentage of Total Population"
        x_formatter = plt.FuncFormatter(lambda x, p: f'{abs(x):.0f}%')
        title_suffix = " (Percentage)"
    elif plot_type == 'population':
        # Calculate common and excess raw populations
        df['Common_Population'] = df[['Male Population', 'Female Population']].min(axis=1)
        df['Excess_Male_Population'] = df['Male Population'] - df['Common_Population']
        df['Excess_Female_Population'] = df['Female Population'] - df['Common_Population']

        male_common_data = df['Common_Population']
        male_excess_data = df['Excess_Male_Population']
        female_common_data = df['Common_Population']
        female_excess_data = df['Excess_Female_Population']

        x_label = "Population"
        x_formatter = plt.FuncFormatter(lambda x, p: f'{abs(x):,.0f}') # Format with commas for population numbers
        title_suffix = " (Raw Population)"
    elif plot_type == 'acc_pct':
        # Calculate common and excess for accumulated percentages
        df['Common_Acc_Percent_Rev'] = df[['Male_Acc_Percent_Rev', 'Female_Acc_Percent_Rev']].min(axis=1)
        df['Excess_Male_Acc_Percent_Rev'] = df['Male_Acc_Percent_Rev'] - df['Common_Acc_Percent_Rev']
        df['Excess_Female_Acc_Percent_Rev'] = df['Female_Acc_Percent_Rev'] - df['Common_Acc_Percent_Rev']

        male_common_data = df['Common_Acc_Percent_Rev']
        male_excess_data = df['Excess_Male_Acc_Percent_Rev']
        female_common_data = df['Common_Acc_Percent_Rev']
        female_excess_data = df['Excess_Female_Acc_Percent_Rev']

        x_label = "Accumulated Percentage of Total Population (Above Age)"
        x_formatter = plt.FuncFormatter(lambda x, p: f'{abs(x):.0f}%')
        title_suffix = " (Accumulated Percentage)"
    else:
        raise ValueError("plot_type must be 'percent', 'population', or 'acc_pct'")

    # --- Debugging: Print numbers being plotted for each age group in a single line ---
    print("\n--- Debug: Data being plotted per age group (single line) ---")
    for i, row in df.iterrows():
        group = row['GROUP']
        original_male_pop = row['Male Population']
        original_female_pop = row['Female Population']

        extra_info = ""
        if plot_type == 'percent':
            if row['Excess_Male_Percent'] > 0:
                extra_info = f", Extra Male {row['Excess_Male_Percent']:.2f}%"
            elif row['Excess_Female_Percent'] > 0:
                extra_info = f", Extra Female {row['Excess_Female_Percent']:.2f}%"
            print(f"Age Group: {group}, Male {original_male_pop:,.0f}, Female {original_female_pop:,.0f}{extra_info}")
        elif plot_type == 'population':
            if row['Excess_Male_Population'] > 0:
                extra_info = f", Extra Male {row['Excess_Male_Population']:,.0f}"
            elif row['Excess_Female_Population'] > 0:
                extra_info = f", Extra Female {row['Excess_Female_Population']:,.0f}"
            print(f"Age Group: {group}, Male {original_male_pop:,.0f}, Female {original_female_pop:,.0f}{extra_info}")
        elif plot_type == 'acc_pct':
            # For acc_pct, print common and excess values for clarity in debug
            male_common_debug = row['Common_Acc_Percent_Rev']
            male_excess_debug = row['Excess_Male_Acc_Percent_Rev']
            female_common_debug = row['Common_Acc_Percent_Rev']
            female_excess_debug = row['Excess_Female_Acc_Percent_Rev']
            
            extra_acc_info = ""
            if male_excess_debug > 0:
                extra_acc_info = f", Extra Male Acc Pct {male_excess_debug:.2f}%"
            elif female_excess_debug > 0:
                extra_acc_info = f", Extra Female Acc Pct {female_excess_debug:.2f}%"
            print(f"Age Group: {group}, Male Acc Pct: {row['Male_Acc_Percent_Rev']:.2f}%, Female Acc Pct: {row['Female_Acc_Percent_Rev']:.2f}%{extra_acc_info}")
    print("----------------------------------------------------")
    # --- End Debugging ---

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    bar_height = 0.8

    # Plot Male population (left side) - Common part
    ax.barh(df['GROUP'], -male_common_data, height=bar_height, color='skyblue', label='Male (Common)')
    # Plot Male population (left side) - Excess part
    ax.barh(df['GROUP'], male_excess_data, height=bar_height, left=-(male_common_data + male_excess_data),
            color='steelblue', label='Male (Excess)')

    # Plot Female population (right side) - Common part
    ax.barh(df['GROUP'], female_common_data, height=bar_height, color='lightcoral', label='Female (Common)')
    # Plot Female population (right side) - Excess part
    ax.barh(df['GROUP'], female_excess_data, height=bar_height, left=female_common_data,
            color='indianred', label='Female (Excess)')
    
    # Custom legend handles for common and excess parts (consistent for all plot types now)
    handles = [
        Patch(facecolor='skyblue', label='Male (Common)'),
        Patch(facecolor='steelblue', label='Male (Excess)'),
        Patch(facecolor='lightcoral', label='Female (Common)'),
        Patch(facecolor='indianred', label='Female (Excess)')
    ]
    
    ax.legend(handles=handles, loc='upper right')

    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_ylabel("Age Group")
    
    # Get the sorted list of age group labels and their corresponding positions
    age_groups_list = df['GROUP'].tolist()
    tick_positions = []
    tick_labels = []

    for i, group_label in enumerate(age_groups_list):
        # Check if the group is '100+' or a digit string that is a multiple of 5
        if group_label == '100+':
            tick_positions.append(i)
            tick_labels.append(group_label)
        elif group_label.isdigit():
            age_num = int(group_label)
            if age_num % 5 == 0:
                tick_positions.append(i)
                tick_labels.append(group_label)

    # Set custom y-axis ticks and labels
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Format x-axis labels dynamically
    ax.xaxis.set_major_formatter(x_formatter)
    
    ax.set_title(f"Bangladesh Population Pyramid 2024{title_suffix}", fontsize=16)
    
    # Add subtitle for data source
    fig.text(0.3, 0.97, "* Data source: US Census International Database (IDB)",
             ha='center', va='center', fontsize=10, color='gray', transform=ax.transAxes)

    # Set x-axis limits symmetrically to ensure the pyramid is centered
    # max_val should be based on the total (common + excess) for the selected plot type
    max_val = max(male_common_data.max() + male_excess_data.max(), 
                  female_common_data.max() + female_excess_data.max()) if not df.empty else 10
    ax.set_xlim(-max_val * 1.1, max_val * 1.1) 

    # --- Add more x-axis labels for percentage plots (every 5%) ---
    if plot_type in ['acc_pct']:
        # Determine the maximum absolute value on the x-axis (excluding padding)
        current_max_x = max(male_common_data.max() + male_excess_data.max(),
                            female_common_data.max() + female_excess_data.max())
        
        # Create ticks at 5% intervals up to and including the max value
        # Ensure 0 is always included
        positive_ticks = np.arange(0, current_max_x + 5, 5)
        # Create symmetric negative ticks
        negative_ticks = -positive_ticks[1:] # Exclude 0 from negative ticks to avoid duplication
        
        # Combine and sort all ticks
        all_ticks = np.sort(np.concatenate((negative_ticks, positive_ticks)))
        
        # Set the x-axis ticks
        ax.set_xticks(all_ticks)
        # The x_formatter will handle the labels correctly (e.g., -5% becomes 5%)
    # --- End of x-axis label additions ---

    ax.grid(axis='x', linestyle='--', alpha=0.7) # Keep vertical grid lines

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the plot with a tight bounding box to prevent labels from being cut off
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Population pyramid plot saved to {output_path}")
    plt.close(fig) # Close the figure to free up memory

def main():
    parser = argparse.ArgumentParser(description="Generate a population pyramid plot.")
    parser.add_argument(
        "csv_file", 
        nargs='?', # Makes the argument optional
        default='./data/IDB_07-05-2025.csv', # Sets the default value
        help="Path to the input CSV file for population pyramid. Defaults to './data/IDB_07-05-2025.csv' if not provided."
    )
    parser.add_argument(
        "--plot_type",
        default="percent",
        choices=["percent", "population", "acc_pct"], # Added 'acc_pct' as a choice
        help="Type of data to plot: 'percent', 'population', or 'acc_pct'. Defaults to 'percent'."
    )
    args = parser.parse_args()
    csv_file = args.csv_file

    OUTPUT_DIR = "plots"
    # Adjust output filename based on plot type
    OUTPUT_FILENAME = f"population_pyramid_{args.plot_type}.png"

    plot_population_pyramid(csv_file, OUTPUT_DIR, OUTPUT_FILENAME, plot_type=args.plot_type)

if __name__ == "__main__":
    main()
