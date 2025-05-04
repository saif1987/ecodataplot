import pandas as pd
import numpy as np
import os
import imageio
import tempfile
import shutil
from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from bokeh.models import NumeralTickFormatter
from bokeh.io import export_png

def create_bokeh_animation_gif(df: pd.DataFrame, years: list[int], output_dir: str, output_filename: str):
    """
    Generates and saves the animated scatter plot using Bokeh frames.

    Args:
        df: The filtered DataFrame ready for plotting.
        years: A sorted list of unique years present in the filtered data.
        output_dir: The directory to save the output GIF.
        output_filename: The name for the output GIF file.
    """
    if df is None or df.empty:
        print("No data available for plotting.")
        return

    output_path = os.path.join(output_dir, output_filename)
    countries = sorted(df['Country'].unique())
    palette = Category10[max(3, min(10, len(countries)))]
    color_map = factor_cmap('Country', palette=palette, factors=countries)

    # Determine axis limits with padding
    min_imp = df['Imports (% GDP)'].min()
    max_imp = df['Imports (% GDP)'].max()
    min_exp = df['Exports (% GDP)'].min()
    max_exp = df['Exports (% GDP)'].max()
    padding_imp = (max_imp - min_imp) * 0.1 if max_imp > min_imp else 1
    padding_exp = (max_exp - min_exp) * 0.1 if max_exp > min_exp else 1

    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory for frames: {temp_dir}")
    png_files = []

    # Initialize plot with data from the first year
    first_year_data = df[df['Year'] == years[0]]
    source = ColumnDataSource(data=first_year_data)

    p = figure(
        height=600, width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        tooltips=[("Country", "@Country"), ("Imports", "@{%Imports (% GDP)}{0.0}%"), ("Exports", "@{%Exports (% GDP)}{0.0}%")],
        x_range=(min_imp - padding_imp, max_imp + padding_imp),
        y_range=(min_exp - padding_exp, max_exp + padding_exp),
        title=f"Imports vs. Exports as % of GDP - Year: {years[0]}"
    )
    p.xaxis.axis_label = "Imports (% of GDP)"
    p.yaxis.axis_label = "Exports (% of GDP)"
    p.xaxis.formatter = NumeralTickFormatter(format="0.0'%'")
    p.yaxis.formatter = NumeralTickFormatter(format="0.0'%'")

    p.scatter(
        x='Imports (% GDP)', y='Exports (% GDP)', source=source,
        legend_field='Country', color=color_map, size=12, alpha=0.7
    )
    p.legend.location = "top_left"
    p.legend.title = "Countries"
    p.legend.click_policy="hide"

    print("Generating frames...")
    for i, year in enumerate(years):
        print(f"  Processing year: {year} ({i+1}/{len(years)})")
        current_data = df[df['Year'] == year]
        source.data = ColumnDataSource.from_df(current_data) # Update source data
        p.title.text = f"Imports vs. Exports as % of GDP - Year: {year}" # Update title

        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        try:
            export_png(p, filename=frame_path)
            png_files.append(frame_path)
        except Exception as e:
            print(f"    Error exporting frame for year {year}: {e}")
            print("    Ensure selenium webdriver (geckodriver/chromedriver) is installed and in PATH.")

    if not png_files:
        print("No frames were generated. Cannot create GIF.")
    else:
        print(f"\nStitching {len(png_files)} frames into GIF: {output_path}...")
        try:
            os.makedirs(output_dir, exist_ok=True)
            with imageio.get_writer(output_path, mode='I', duration=0.2, loop=0) as writer:
                for filename in png_files:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print("GIF saved successfully.")
        except Exception as e:
            print(f"Error creating GIF: {e}")

    try:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")