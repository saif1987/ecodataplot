import pandas as pd
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import PercentFormatter # For formatting axes as percentages

def create_seaborn_animation(df: pd.DataFrame, years: list[int], output_dir: str, output_filename: str):
    """
    Generates and saves an animated scatter plot using Seaborn/Matplotlib.

    Args:
        df: The filtered DataFrame ready for plotting.
        years: A sorted list of unique years present in the filtered data.
        output_dir: The directory to save the output animation.
        output_filename: The name for the output animation file (e.g., .gif).
    """
    if df is None or df.empty:
        print("No data available for interactive plot.")
        return

    output_path = os.path.join(output_dir, output_filename)
    countries = sorted(df['Country'].unique())

    # --- Setup Plot ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    min_imp = df['Imports (% GDP)'].min()
    max_imp = df['Imports (% GDP)'].max()
    min_exp = df['Exports (% GDP)'].min()
    max_exp = df['Exports (% GDP)'].max()
    padding_imp = (max_imp - min_imp) * 0.1 if max_imp > min_imp else 1
    padding_exp = (max_exp - min_exp) * 0.1 if max_exp > min_exp else 1

    ax.set_xlim((min_imp - padding_imp), (max_imp + padding_imp))
    ax.set_ylim((min_exp - padding_exp), (max_exp + padding_exp))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xlabel("Imports (% of GDP)")
    ax.set_ylabel("Exports (% of GDP)")
    fig.suptitle("Imports vs. Exports as % of GDP Over Time", fontsize=16)

    # Use seaborn's color palette
    palette = sns.color_palette("tab10", n_colors=len(countries))
    color_dict = dict(zip(countries, palette))

    # Initial scatter plot (empty) & year text
    scatter = ax.scatter([], [], s=100, alpha=0.7) # s is marker size
    year_text = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right', fontsize=14, weight='bold')

    # Create legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=country,
                                 markerfacecolor=color_dict[country], markersize=10)
                      for country in countries]
    ax.legend(handles=legend_handles, title="Countries", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust layout for legend

    # --- Animation Function ---
    def update(year):
        """Update function for Matplotlib animation."""
        current_data = df[df['Year'] == year]
        if current_data.empty:
            scatter.set_offsets(np.empty((0, 2))) # Clear points if no data
            scatter.set_facecolor(np.empty((0, 4)))
        else:
            offsets = current_data[['Imports (% GDP)', 'Exports (% GDP)']].values
            colors = [color_dict[country] for country in current_data['Country']]
            scatter.set_offsets(offsets)
            scatter.set_facecolor(colors)

        year_text.set_text(str(year))
        ax.set_title(f"Year: {year}")
        print(f"  Processing frame for year: {year}") # Progress indicator
        return scatter, year_text

    # --- Create and Save Animation ---
    print("Creating animation...")
    ani = animation.FuncAnimation(fig, update, frames=years, interval=200, blit=True, repeat=False)

    print(f"Saving animation to: {output_path}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Use Pillow writer for GIF (usually available with matplotlib)
        ani.save(output_path, writer='pillow', fps=5)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure 'pillow' is installed. You might need 'imagemagick' for other formats.")

    plt.close(fig) # Close the plot figure after saving