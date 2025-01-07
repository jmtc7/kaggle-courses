import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../data/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../data/home-data-for-ml-course/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data and print first elements of trainig set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
print('Sample of training data:')
print(X_train.head())
print('\n')

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

min_err = 999999
min_err_idx = -1
for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
    if mae < min_err:
        min_err = mae
        min_err_idx = i

print(f'\nThe best model is the number {min_err_idx}, with a MAE of {min_err}')

# Build and train final model with all training set
final_model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
final_model.fit(X, y)

# Generate test predictions and generate submission CSV for Kaggle competition
preds_test = final_model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('00submission.csv', index=False)
