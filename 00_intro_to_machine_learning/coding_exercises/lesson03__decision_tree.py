from sklearn.tree import DecisionTreeRegressor # To build a REGRESSION Decision Tree
import pandas as pd # To manage data

# Load CSV with Melbourne prices as a DataFrame
# Downloaded from https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
melbourne_prices_csv_path = '../data/melb_data.csv'
melbourne_df = pd.read_csv(melbourne_prices_csv_path)

# Print columns (variables) and drop missing values
print(melbourne_df.columns)
melbourne_df = melbourne_df.dropna(axis=0)

# Select the prediction target and input features
y = melbourne_df.Price
x = melbourne_df[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]

# Get statistics about the chosen features and show the first few ones
print('\n Statistics about the features:')
print(x.describe())
print('\n First 5 features:')
print(x.head())
print('\n First 5 prices:')
print(y.head())

# Define a decision tree model
# The random_state is specified to ensure repetitiviness
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model using the selected features and prediction target
melbourne_model.fit(x, y)

# Predict the price for the first houses
print("\nThe predictions for the first 5 houses are:")
print(melbourne_model.predict(x.head()))