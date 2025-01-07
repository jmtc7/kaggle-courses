from sklearn.model_selection import train_test_split # To split data for training and validation
from sklearn.metrics import mean_absolute_error # To compute MAE metric
from sklearn.tree import DecisionTreeRegressor # To build a regression decision tree
import pandas as pd # To manage data

# Load CSV with Melbourne prices as a DataFrame and drop missing values
# Downloaded from https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
melbourne_prices_csv_path = '../../data/melbourne-housing-snapshot/melb_data.csv'
melbourne_df = pd.read_csv(melbourne_prices_csv_path)
melbourne_df = melbourne_df.dropna(axis=0)

# Select the prediction target and input features
y = melbourne_df.Price
x = melbourne_df[['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']]

# Split the data into training and validation subsets
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

# Define and fit a regression decision tree model
# random_state for repetitiviness
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(train_x, train_y)

# Predict with the model and compute MAE
val_predictions = melbourne_model.predict(val_x)
val_mae = mean_absolute_error(val_y, val_predictions)
print(f'Computer Mean Absolute Error: {val_mae}')
