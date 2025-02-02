import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def score_dataset(X_t, X_v, y_t, y_v):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

# Load the data
data = pd.read_csv('../../data/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Get insigths about missing data
missing_val_count_by_column = X_train.isnull().sum()
cols_w_missing_data = missing_val_count_by_column[missing_val_count_by_column > 0].count()
print(f'Rows in the training data: {X_train.shape[0]}')
print(f'Cols in the training data with missing values: {cols_w_missing_data}')
print(f'Totla missing entries in the training data: {missing_val_count_by_column.values.sum()}')
print()

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
print(f'Cols with missing values: {cols_with_missing}')
print()

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
print()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train)) # Fit and transform training data
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid)) # No need to fit data since it was done for the training set

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
print()

# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
print()

# Shape of training data (num_rows, num_columns)
print(f'Shape of training data: {X_train.shape}')

# Number of missing values in each column of training data
print('Missing values in each column of the training data:')
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
print()

# Load Kaggle Housing Prices competition data
X_full = pd.read_csv('../../data/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../../data/home-data-for-ml-course/test.csv', index_col='Id')

# Remove samples with missing prices and create prediction target for training
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
final_y_train = X_full.SalePrice

# Drop prediction target (house prices) and numerical predictors
X_full.drop(['SalePrice'], axis=1, inplace=True)
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Perform imputation for missing values
my_imputer = SimpleImputer()
final_X_train = pd.DataFrame(my_imputer.fit_transform(X))
final_X_test = pd.DataFrame(my_imputer.transform(X_test))

# Build and train final model with all training set
final_model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
final_model.fit(final_X_train, final_y_train)

# Generate test predictions and generate submission CSV for Kaggle competition
preds_test = final_model.predict(final_X_test)
output = pd.DataFrame({'Id': final_X_test.index+1461, 'SalePrice': preds_test})
output.to_csv('lesson02__submission.csv', index=False)
