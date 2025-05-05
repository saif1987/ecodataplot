# Economic Data Animation Project

## Overview

This project aims to visualize the economic development of countries over the past 50 years (1974-2024) through animated plots. By collecting and analyzing key economic indicators such as GDP, Exports, Imports, and Population, we can gain insights into how these factors have evolved and interacted over time.

## Project Steps

1.  **Data Collection:**

    * Collect economic data for all countries, spanning the last 50 years.
**
    * Key indicators include:

        * GDP (Gross Domestic Product)

        * Exports

        * Imports

        * Population

    * Data source:

        * World Bank Indicator API

2.  **Animated Plotting Script:**

    * Develop a Python script to generate professional-level animated plots.

    * Utilize a suitable Python visualization library (e.g., Bokeh) for creating interactive and dynamic animations.

3.  **Animated Visualizations:**

    * Create time-animated plots to visualize the relationships between economic indicators:

        * GDP vs. Exports

        * GDP vs. Imports

        * Other relevant combinations

##   Data Source

The following data source will be used:

* **World Bank Indicator API:** For comprehensive economic data, particularly the World Development Indicators.

##   Python Libraries

The following Python libraries will be used:

* **Pandas:** For data manipulation, cleaning, and analysis.

* **Bokeh:** For creating interactive and animated plots.

* **Other libraries:** (e.g., requests, wbgapi) for interacting with the World Bank Indicator API.

##   Project Structure

    ├── data/          # Directory for storing raw and processed data
    ├── data_collection/ # Directory for data collection scripts
    │   └── data_collection.py  # Script for collecting data
    ├── scripts/       # Directory for plotting logic and execution
    │   ├── data_loader.py    # Loads data from CSV
    │   ├── data_processor.py # Processes and filters data
    │   ├── plotter.py        # Generates the animation plot
    │   ├── main.py           # Main script to run plotting pipeline
    │   └── fetch_flags.py    # Script to download flag icons
    ├── notebooks/     # Directory for Jupyter Notebooks (exploration, analysis)
    ├── tests/         # Directory for unit tests
    ├── README.md      # Project README file
    ├── requirements.txt # Project dependencies

##  Getting Started

1.  **Clone the repository:**

        git clone <repository_url>
        cd economic-data-animation

2.  **Set up a virtual environment (recommended):**

        python3 -m venv venv
        source venv/bin/activate  # On Linux/macOS
        venv\Scripts\activate  # On Windows

3.  **Install the required packages:**

        pip install -r requirements.txt

4.  **Obtain data:**

    * Run the data collection script to fetch data from the World Bank Indicator API.
        ```bash
        python data_collection/data_collection.py
        ```
    * This creates a CSV file in the `data/` directory.

5.  **Fetch Flag Icons (Run once):**
    * Run the flag fetching script to download icons needed for the plot.
        ```bash
        python scripts/fetch_flags.py
        ```
    * This creates rounded flag icons in the `icons/` directory.

6.  **Generate the Animation:**
    * Run the main plotting script, providing the path to the CSV file generated in step 4.
        ```bash
        # Example: Replace with the actual name of your data file
        python scripts/main.py data/economic_indicators_1975-2025_20250504.csv
        ```

##  Visualizations

The project will generate animated plots, including:

* **GDP vs. Exports:** This animation will show how a country's exports change in relation to its GDP over the years.

* **GDP vs. Imports:** This animation will illustrate the relationship between a country's imports and its GDP.

* *(Additional plots can be added)*

## Contributions

Contributions are welcome! If you have any suggestions, find any issues, or would like to contribute to the project, please feel free to submit a pull request or open an issue.
