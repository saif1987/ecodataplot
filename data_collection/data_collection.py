import wbgapi as wb
import pandas as pd # Import pandas for DataFrame handling
import time         # Import time for adding delays
import datetime     # Import datetime to get the current year
import os           # Import os for path manipulation

def get_all_countries():
    """
    Fetches a list of all economies (countries and aggregates)
    from the World Bank API.
    """
    try:
        countries = wb.economy.list()
        return countries
    except Exception as e:
        print(f"Error fetching country list: {e}")
        return []

def get_economic_indicators(economy_ids, time_period, batch_size=50, delay_seconds=0.5):
    """
    Fetches GDP, Population, Poverty Rate, Imports, Exports, and GNI per capita data
    for a list of economy IDs for a specific time period, handling potential
    URL length limits
    by batching requests.

    Args:
        economy_ids (list): A list of World Bank economy IDs.
        time_period (range, list, str): The time period to fetch data for
                                         (e.g., range(2000, 2020)).
        batch_size (int): The number of economies to request in each API call.

    Returns:
        pandas.DataFrame: A DataFrame containing the requested data,
                          or an empty DataFrame if an error occurs.
    """
    indicators = {
        'NY.GDP.MKTP.CD': 'GDP (Current US$)',
        'SP.POP.TOTL': 'Population',
        'SI.POV.DDAY': 'Poverty Rate ($2.15/day, %)',
        'NE.IMP.GNFS.CD': 'Imports (Current US$)', # Added Imports
        'NE.EXP.GNFS.CD': 'Exports (Current US$)', # Added Exports
        'NY.GNP.PCAP.CD': 'GNI per capita (Current US$)' # Added GNI per capita
    }
    all_data_dfs = [] # List to store DataFrames from each batch

    try:
        print(f"\nFetching Economic Indicators for time period {time_period} in batches...")
        # Process economies in batches
        for i in range(0, len(economy_ids), batch_size):
            batch_ids = economy_ids[i:i + batch_size]
            print(f"  Fetching batch {i//batch_size + 1} ({len(batch_ids)} economies)...")
            try:
                # Use labels=True to include economy names directly
                batch_df = wb.data.DataFrame(indicators.keys(), batch_ids, time=time_period, labels=True)
                # Rename columns for clarity
                batch_df = batch_df.rename(columns=indicators)
                all_data_dfs.append(batch_df)
            except Exception as batch_e:
                print(f"  Error fetching batch starting with {batch_ids[0]}: {batch_e}")
                # Decide if you want to skip the batch or stop entirely
                # continue # Uncomment to skip failed batches
                # return pd.DataFrame() # Uncomment to stop on first batch error
            finally:
                # Add a small delay after each batch request (successful or not)
                time.sleep(delay_seconds)

        # Concatenate all collected DataFrames
        if all_data_dfs:
            final_df = pd.concat(all_data_dfs)
            return final_df
        else:
            return pd.DataFrame() # Return empty if no batches succeeded

    except Exception as e:
        print(f"Error fetching indicator data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

if __name__ == "__main__":
    print("Fetching list of economies from World Bank API...")
    all_countries = get_all_countries()

    # Define the time range for the last 50 years
    current_year = datetime.datetime.now().year
    start_year = current_year - 50
    collection_date = datetime.date.today().strftime('%Y%m%d') # Format date as YYYYMMDD
    target_time_period = range(start_year, current_year + 1) # +1 to include the current year

    # Define output path
    output_dir = "data"
    output_filename = f"economic_indicators_{start_year}-{current_year}_{collection_date}.csv"
    if all_countries:
        economy_ids = [country['id'] for country in all_countries] # Extract IDs
        economic_data = get_economic_indicators(economy_ids, time_period=target_time_period) # Fetch data

        if not economic_data.empty:
            print(f"\n--- Economic Indicators ({start_year}-{current_year}) ---")
            # Configure pandas to display more rows/columns if needed
            # pd.set_option('display.max_rows', None) # Show all rows
            # pd.set_option('display.max_columns', None) # Show all columns
            # print(economic_data) # Replaced printing with saving

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)

            # Save the DataFrame to CSV
            print(f"Saving data to {output_path}...")
            economic_data.to_csv(output_path, index=True) # Keep multi-index (economy, time)
            print("Data saved successfully.")
        else:
            print("Could not retrieve the requested economic indicators.")
    else:
        print("Could not retrieve country list.")