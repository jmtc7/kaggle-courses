import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Read the data and separate target from predictors
#data = pd.read_csv('C:/Users/uic84925/Downloads/kaggle-courses/data/melbourne-housing-snapshot/melb_data.csv')
data = pd.read_csv('../../data/melbourne-housing-snapshot/melb_data.csv')
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# ANALYZE DATA -------------------------------------------------------------------------------------

# Get list of numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
print(f'Numerical variables:\n  {numerical_cols}')

# Get list of categorical variables
object_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == "object"]
print(f'Categorical variables: ')

# Get lists of categorical variables with high and low cardinality (more/less than 10 values)
low_card_thresh = 10
low_cardinality_cols = [col for col in object_cols if X_train_full[col].nunique() < low_card_thresh]
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))
print(f'  - Low cardinality (less than 10 different values): {low_cardinality_cols}')
print(f'  - High cardinality (more than 10 different values): {high_cardinality_cols}')

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train_full[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
print('\nNumber of unique values in each categorical variable:')
print(sorted(d.items(), key=lambda x: x[1]))
high_cardinality_numcols = len([x for x in d.keys() if d[x]>10])
print(f'Number of categorical variables with cardinality > 10: {high_cardinality_numcols}')
print(f'Number of different values in variable "Method": {d["Method"]}')

# PREPARE DATA -------------------------------------------------------------------------------------

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Overwrite list of categorical variables, since we are now only using the low cardinality ones
object_cols = low_cardinality_cols

# Analyze issues with values appearing in validation but not training
# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print(f'\nCategorical columns whose values are in both training and validation sets: {good_label_cols}')
print(f'Categorical columns with values in validation set that are not in the training set: {bad_label_cols}\n')

# OPTION 1: DROP CATEGORICAL VARIABLES -------------------------------------------------------------
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


# OPTION 2: ORDINAL ENCODING -----------------------------------------------------------------------
# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply (random) ordinal encoder to each column with categorical data
# NOTE: It could be improved with custom integers to sort the ordinal values.
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


# OPTION 3: ONE-HOT ENCODING -----------------------------------------------------------------------
# Apply one-hot encoder to each column with categorical data
# NOTE: handle_unknown='ignore' to avoid errors when validation data contains classes that aren't in the training data
# NOTE: sparse=False to return the encoded columns as a numpy array (instead of a sparse matrix)
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
