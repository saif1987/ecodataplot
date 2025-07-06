import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


def plot_population_pyramid(csv_path: str, output_dir: str, output_filename: str):
    """
    Generates and saves a standard population pyramid plot.

    Args:
        csv_path: Path to the CSV file containing population data.
                  Expected columns: 'GROUP' (age group), 'Male Population', 'Female Population'.
        output_dir: Directory to save the output plot.
        output_filename: Name for the output plot file (e.g., .png).
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

    # Filter out rows where 'GROUP' is an empty string, as these are often summary rows
    # and not specific age groups for the pyramid.
    df = df[df['GROUP'].str.strip() != '']
    df = df[df['GROUP'].str.strip().str.lower() != 'nan'] # Added this line to filter out 'nan' strings
    df = df[df['GROUP'].str.strip().str.lower() != '100+'] # Added this line to filter out 'nan' strings


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
    
    # --- DEBUG LINE ADDED HERE ---
    print(f"\n--- Debug: Unique GROUP values after filtering and sorting: {df['GROUP'].unique()} ---")
    print("----------------------------------------------------")
    # --- END DEBUG LINE ---

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
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    bar_height = 0.8

    # Plot Male population (left side)
    # Use negative values for male percentages to plot them to the left of the central axis
    ax.barh(df['GROUP'], -df['Male_Percent'], height=bar_height, color='skyblue', label='Male')

    # Plot Female population (right side)
    ax.barh(df['GROUP'], df['Female_Percent'], height=bar_height, color='lightcoral', label='Female')

    # Customize plot
    ax.set_xlabel("Percentage of Total Population")
    ax.set_ylabel("Age Group")
    
    # --- Start of Y-axis Label and Grid Line Changes ---
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
    # --- End of Y-axis Label and Grid Line Changes ---

    # Format x-axis labels to show absolute percentage values
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{abs(x):.0f}%'))
    
    ax.set_title("Population Pyramid", fontsize=16)

    # Set x-axis limits symmetrically to ensure the pyramid is centered
    # Add some padding (10%) for better visualization
    max_percent = max(df['Male_Percent'].max(), df['Female_Percent'].max()) if not df.empty else 10
    ax.set_xlim(-max_percent * 1.1, max_percent * 1.1) 

    ax.legend()
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
    args = parser.parse_args()
    csv_file = args.csv_file

    OUTPUT_DIR = "plots"
    OUTPUT_FILENAME = "population_pyramid.png"

    plot_population_pyramid(csv_file, OUTPUT_DIR, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
