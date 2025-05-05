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
    Generates and saves an animated scatter plot using Seaborn/Matplotlib,
    with smooth interpolation between yearly points.

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
    INTERPOLATION_STEPS = 5 # Number of intermediate frames between years (adjust for smoother/faster)
    countries = sorted(df['Country'].unique())

    # --- Setup Plot ---
    # sns.set_theme(style="whitegrid") # Remove or comment out seaborn theme
    plt.style.use('dark_background') # Use matplotlib's dark background style
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
    palette = sns.color_palette("bright", n_colors=len(countries)) # Try 'bright' palette for more pop
    color_dict = dict(zip(countries, palette))

    # Initial scatter plot (empty) & year text
    scatter = ax.scatter([], [], s=120, alpha=0.85) # Slightly larger size and higher alpha
    year_text = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right', fontsize=14, weight='bold')

    # Create legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=country,
                                 markerfacecolor=color_dict[country], markersize=10)
                      for country in countries]
    ax.legend(handles=legend_handles, title="Countries", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust layout for legend

    # --- Prepare Data for Interpolation ---
    # Create a lookup dictionary: (country, year) -> (imp%, exp%)
    coord_dict = {}
    df_indexed = df.set_index(['Country', 'Year'])
    for country in countries:
        for year in years:
            if (country, year) in df_indexed.index:
                imp_exp = df_indexed.loc[(country, year), ['Imports (% GDP)', 'Exports (% GDP)']].values
                if not np.isnan(imp_exp).any():
                    coord_dict[(country, year)] = imp_exp

    # --- Animation Function ---
    def update(frame_info):
        """Update function for Matplotlib animation."""
        year1, year2, step_frac = frame_info
        current_offsets = []
        current_colors = []

        for country in countries:
            pos1 = coord_dict.get((country, year1))
            pos2 = coord_dict.get((country, year2))

            # Determine position based on availability and interpolation fraction
            current_pos = None
            if pos1 is not None and pos2 is not None:
                current_pos = pos1 + (pos2 - pos1) * step_frac
            elif pos1 is not None and step_frac < 0.5: # Show pos1 if pos2 missing, near start
                current_pos = pos1
            elif pos2 is not None and step_frac >= 0.5: # Show pos2 if pos1 missing, near end
                current_pos = pos2

            if current_pos is not None:
                current_offsets.append(current_pos)
                current_colors.append(color_dict[country])

        if not current_offsets:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_facecolor(np.empty((0, 4)))
        else:
            scatter.set_offsets(np.array(current_offsets))
            scatter.set_facecolor(current_colors)

        # Display the starting year of the transition in the title/text
        year_display = year1
        year_text.set_text(str(year_display))
        ax.set_title(f"Year: {year_display}")
        print(f"  Processing frame for year: {year1} (step {step_frac:.2f})") # Progress indicator
        return scatter, year_text

    # --- Create and Save Animation ---
    # Generate frames including intermediate steps
    animation_frames = []
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i+1]
        for step in range(INTERPOLATION_STEPS):
            frac = step / INTERPOLATION_STEPS
            animation_frames.append((y1, y2, frac))
    # Add the final year's frame
    if years:
        animation_frames.append((years[-1], years[-1], 0.0)) # Show final frame distinctly

    print(f"Creating animation with {len(animation_frames)} frames...")
    # Adjust interval based on desired speed and interpolation steps
    # E.g., if original interval was 500ms (fps=2), new interval is 500/INTERPOLATION_STEPS
    interval_ms = max(20, 500 // INTERPOLATION_STEPS) # Ensure interval is at least 20ms
    ani = animation.FuncAnimation(fig, update, frames=animation_frames, interval=interval_ms, blit=True, repeat=False)

    print(f"Saving animation to: {output_path}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Use Pillow writer for GIF (usually available with matplotlib)
        # Adjust fps based on the new interval
        save_fps = 1000 // interval_ms
        ani.save(output_path, writer='pillow', fps=save_fps)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure 'pillow' is installed. You might need 'imagemagick' for other formats.")

    plt.close(fig) # Close the plot figure after saving
