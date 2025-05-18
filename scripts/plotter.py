import pandas as pd
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import PercentFormatter # For formatting axes as percentages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # For plotting images

def create_seaborn_animation(df: pd.DataFrame, years: list[int], output_dir: str, output_filename: str):
    """
    Generates and saves an animated scatter plot using Seaborn/Matplotlib,
    with smooth interpolation between yearly points.

    Args:
        df: The filtered DataFrame ready for plotting.
        years: A sorted list of unique years present in the filtered data.
        output_dir: The directory to save the output animation.
        output_filename: The name for the output animation file (e.g., .gif).
    """
    if df is None or df.empty:
        print("No data available for trade vs GDP plot.")
        return

    output_path = os.path.join(output_dir, output_filename)
    INTERPOLATION_STEPS = 5
    countries = sorted(df['Country'].unique())
    ICON_DIR = "./icons" # Path to the icons folder relative to this script

    # --- Load Icons ---
    flag_images = {}
    print("Loading flag icons...")
    for country in countries:
        icon_path = os.path.join(ICON_DIR, f"{country}.png")
        try:
            flag_images[country] = plt.imread(icon_path)
        except FileNotFoundError:
            print(f"  Warning: Icon not found for {country} at {icon_path}. Skipping.")
            # Optionally use a placeholder image here
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

    # Remove initial scatter plot
    # scatter = ax.scatter([], [], s=120, alpha=0.85)
    # Store AnnotationBbox objects
    artists = {}
    year_text = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right', fontsize=14, weight='bold')

    # Remove standard legend creation
    # # Create legend handles
    # legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=country,
    #                              markerfacecolor=color_dict[country], markersize=10)
    #                   for country in countries]
    # ax.legend(handles=legend_handles, title="Countries", bbox_to_anchor=(1.05, 1), loc='upper left')

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
        # Clear previous artists managed by this function
        for artist in artists.values():
            artist.remove()
        artists.clear()

        year1, year2, step_frac = frame_info
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

            # Plot image if position and icon are valid
            if current_pos is not None and country in flag_images:
                img = flag_images[country]
                # Adjust zoom based on desired icon size on plot
                imagebox = OffsetImage(img, zoom=0.4) # Adjust zoom factor as needed
                ab = AnnotationBbox(imagebox, current_pos, frameon=False, pad=0)
                artists[country] = ax.add_artist(ab) # Add artist to axes and store reference


        # Display the starting year of the transition in the title/text
        year_display = year1
        year_text.set_text(str(year_display))
        ax.set_title(f"Year: {year_display}")
        print(f"  Processing frame for year: {year1} (step {step_frac:.2f})") # Progress indicator

        # Return list of artists updated in this frame (needed for blit=True, but complex with AnnotationBbox)
        # Setting blit=False simplifies this, as we don't need to return artists.
        # return list(artists.values()) + [year_text] # This might be needed if blit=True works

        return [year_text] # Return only year_text if blit=False

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

    # --- Add Manual Legend (Static - outside animation loop) ---
    legend_x_start = 1.02 # X position relative to axes (adjust as needed)
    legend_y_start = 0.95 # Y position relative to axes (adjust as needed)
    legend_y_step = 0.05   # Vertical spacing between entries (adjust as needed)
    legend_icon_zoom = 0.2 # Zoom factor for legend icons (adjust as needed)

    for i, country in enumerate(countries):
        if country in flag_images:
            y_pos = legend_y_start - i * legend_y_step
            img = flag_images[country]
            imagebox = OffsetImage(img, zoom=legend_icon_zoom)
            ab = AnnotationBbox(imagebox, (legend_x_start, y_pos), xycoords='axes fraction', frameon=False, box_alignment=(0, 0.5))
            ax.add_artist(ab) # Add legend artist once
            ax.text(legend_x_start + 0.04, y_pos, country, transform=ax.transAxes, va='center', ha='left', fontsize=9) # Add legend text once

    print(f"Creating animation with {len(animation_frames)} frames...")
    # Adjust interval based on desired speed and interpolation steps
    # E.g., if original interval was 500ms (fps=2), new interval is 500/INTERPOLATION_STEPS
    interval_ms = max(20, 500 // INTERPOLATION_STEPS) # Ensure interval is at least 20ms
    # Set blit=False as managing AnnotationBbox with blitting can be complex/buggy
    ani = animation.FuncAnimation(fig, update, frames=animation_frames, interval=interval_ms, blit=False, repeat=False)

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

INTERPOLATION_STEPS_GDP_PC = 10 # Number of steps for smooth transition between years

def create_gdp_per_capita_animation(df: pd.DataFrame, years: list[int], output_dir: str, output_filename: str):
    """
    Generates and saves an animated line plot of GDP per capita vs. Year.

    Args:
        df: DataFrame with 'Country', 'Year', 'GDP per capita'.
            Assumed to be interpolated and NaN-free for relevant data points by the caller.
        years: A sorted list of unique years present in the data for animation.
        output_dir: Directory to save the animation.
        output_filename: Filename for the animation (e.g., .gif).
    """
    if df is None or df.empty:
        print("No data available for GDP per capita plot.")
        return

    output_path = os.path.join(output_dir, output_filename)
    countries = sorted(df['Country'].unique())
    ICON_DIR = "./icons" 
    FLAG_ICON_ZOOM_ON_LINE = 0.25 # Zoom factor for flag icons on the line

    # --- Load Icons (for legend) ---
    flag_images = {}
    print("Loading flag icons for GDP per capita plot legend...")
    for country in countries:
        icon_path = os.path.join(ICON_DIR, f"{country}.png")
        try:
            flag_images[country] = plt.imread(icon_path)
        except FileNotFoundError:
            print(f"  Warning: Icon not found for {country} at {icon_path}. Text legend will be used for this country.")

    # --- Prepare Data for Interpolation ---
    # Create a lookup dictionary: (country, year) -> gdp_per_capita
    gdp_pc_data_dict = {}
    df_indexed = df.set_index(['Country', 'Year'])
    for country in countries:
        for year in years:
            if (country, year) in df_indexed.index:
                gdp_pc_data_dict[(country, year)] = df_indexed.loc[(country, year), 'GDP per capita']

    # --- Setup Plot ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))

    min_gdp_pc = df['GDP per capita'].min()
    max_gdp_pc = df['GDP per capita'].max()
    padding_gdp_pc = (max_gdp_pc - min_gdp_pc) * 0.1 if max_gdp_pc > min_gdp_pc and not np.isnan(min_gdp_pc) and not np.isnan(max_gdp_pc) else 100

    ax.set_xlim(min(years), max(years))
    ax.set_ylim(max(0, (min_gdp_pc - padding_gdp_pc) if not np.isnan(min_gdp_pc) else 0), 
                (max_gdp_pc + padding_gdp_pc) if not np.isnan(max_gdp_pc) else 1000)

    ax.set_xlabel("Year")
    ax.set_ylabel("GDP per Capita (current US$)")
    fig.suptitle("GDP per Capita Over Time", fontsize=16)

    palette = sns.color_palette("bright", n_colors=len(countries))
    color_dict = dict(zip(countries, palette))

    lines = {}
    for country in countries:
        lines[country], = ax.plot([], [], lw=2.5, label=country, color=color_dict[country], alpha=0.8)

    # Store AnnotationBbox objects for moving flags on lines
    flag_artists_on_line = {}

    year_text = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right', fontsize=14, weight='bold', color='white')

    plt.tight_layout(rect=[0, 0, 0.83, 0.95]) # Adjust layout for legend

    # --- Animation Function ---
    def update(frame_info):
        year1_idx, year2_idx, step_frac = frame_info
        year1 = years[year1_idx]
        year2 = years[year2_idx]

        # Clear previous flag artists on lines
        for artist in flag_artists_on_line.values():
            artist.remove()
        flag_artists_on_line.clear()

        current_display_year_float = year1 + (year2 - year1) * step_frac
        
        artists_to_return = []

        for country in countries:
            # Get historical data up to year1
            historical_data = df[(df['Country'] == country) & (df['Year'] <= year1)]
            
            x_data = list(historical_data['Year'])
            y_data = list(historical_data['GDP per capita'])

            gdp_pc_y1 = gdp_pc_data_dict.get((country, year1))
            gdp_pc_y2 = gdp_pc_data_dict.get((country, year2))

            current_head_x = current_display_year_float
            current_head_y = None

            if gdp_pc_y1 is not None and gdp_pc_y2 is not None and year1 != year2:
                current_head_y = gdp_pc_y1 + (gdp_pc_y2 - gdp_pc_y1) * step_frac
                # Ensure the line extends to the interpolated point if it's beyond year1
                if not x_data or x_data[-1] < current_head_x:
                     x_data.append(current_head_x)
                     y_data.append(current_head_y)
                elif x_data: # If current_head_x is same as last x_data, update y
                    x_data[-1] = current_head_x
                    y_data[-1] = current_head_y

            elif gdp_pc_y1 is not None: # At year1 or if year2 data is missing
                current_head_x = year1
                current_head_y = gdp_pc_y1
                if not x_data or x_data[-1] != year1 : # Ensure year1 point is included
                    x_data.append(year1)
                    y_data.append(gdp_pc_y1)
                elif x_data: # Update last point if it's year1
                    y_data[-1] = gdp_pc_y1

            lines[country].set_data(x_data, y_data)
            artists_to_return.append(lines[country])

            # Add moving flag icon at the head of the line
            if current_head_y is not None and country in flag_images:
                img = flag_images[country]
                imagebox = OffsetImage(img, zoom=FLAG_ICON_ZOOM_ON_LINE)
                ab = AnnotationBbox(imagebox, (current_head_x, current_head_y), frameon=False, pad=0)
                flag_artists_on_line[(country, 'line_head')] = ax.add_artist(ab)
                artists_to_return.append(ab)
        
        year_text.set_text(str(int(current_display_year_float)))
        ax.set_title(f"GDP per Capita: {int(current_display_year_float)}", color='white')
        print(f"  Processing GDP per capita frame: Year {year1}-{year2} (step {step_frac:.2f})")
        
        artists_to_return.append(year_text)
        return artists_to_return

    # --- Generate Animation Frames ---
    animation_frames = []
    if len(years) > 1:
        for i in range(len(years) - 1):
            for step in range(INTERPOLATION_STEPS_GDP_PC):
                frac = step / INTERPOLATION_STEPS_GDP_PC
                animation_frames.append((i, i + 1, frac))
        # Add the final year's frame distinctly
        animation_frames.append((len(years) - 1, len(years) - 1, 0.0))
    elif len(years) == 1: # Single year data
        animation_frames.append((0, 0, 0.0))

    # --- Add Manual Legend (Static - outside animation loop) ---
    legend_x_start = 1.02 
    legend_y_start = 0.95 
    legend_y_step = 0.045   
    legend_icon_zoom_legend = 0.2 # Different zoom for legend

    for i, country in enumerate(countries):
        y_pos = legend_y_start - i * legend_y_step
        text_x_offset = legend_x_start + 0.01 
        if country in flag_images:
            img = flag_images[country]
            imagebox = OffsetImage(img, zoom=legend_icon_zoom_legend)
            ab = AnnotationBbox(imagebox, (legend_x_start, y_pos), xycoords='axes fraction', frameon=False, box_alignment=(0, 0.5))
            ax.add_artist(ab)
            text_x_offset = legend_x_start + 0.04 
        
        ax.text(text_x_offset, y_pos, country, transform=ax.transAxes, 
                va='center', ha='left', fontsize=9, color=color_dict[country])

    if not animation_frames:
        print("No frames to animate for GDP per capita plot. Skipping animation generation.")
        plt.close(fig)
        return

    num_frames = len(animation_frames)
    # Adjust interval based on desired speed and interpolation steps
    interval_ms = max(20, 400 // INTERPOLATION_STEPS_GDP_PC) # e.g. 400ms per year / steps
    # Set blit=False as managing AnnotationBbox with blitting can be complex
    ani = animation.FuncAnimation(fig, update, frames=animation_frames, interval=interval_ms, blit=False, repeat=False)

    print(f"Saving GDP per capita animation to: {output_path}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        save_fps = 1000 // interval_ms
        ani.save(output_path, writer='pillow', fps=save_fps)
        print("GDP per capita animation saved successfully.")
    except Exception as e:
        print(f"Error saving GDP per capita animation: {e}")
        print("Ensure 'pillow' is installed.")
    plt.close(fig)
