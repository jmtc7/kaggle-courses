import pandas as pd

# Path to the CSV with the Melbourne housing prices
# Downloaded from https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
melbourne_prices_csv_path = '../../data/melbourne-housing-snapshot/melb_data.csv'

# Read data and store in DataFrame
melbourne_df = pd.read_csv(melbourne_prices_csv_path) 

# Print summary of Melbourne data
print(melbourne_df.describe()) # Print 8 lines for each column

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = round(melbourne_df['Landsize'].mean())
print(avg_lot_size)

# As of today, how old is the newest home (current year - the date in which it was built)
non_nan_yearbuilts = melbourne_df['YearBuilt'].dropna() # Drop "nan" values from YearBuilt series
home_ages = [2024 - year_built for year_built in non_nan_yearbuilts]
newest_home_age = min(home_ages)
print(newest_home_age)
