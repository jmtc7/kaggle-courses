import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

PATH_TO_SUBMISSION_CSV = 'lesson03__submission.csv'
CARDENALITY_THRESHOLD = 10
PATH_TO_TRAIN_CSV = '../../data/home-data-for-ml-course/train.csv'
PATH_TO_TEST_CSV = '../../data/home-data-for-ml-course/test.csv'
TARGET_VAR = 'SalePrice'
N_TREES = 100

def split_X_y(X: pd.DataFrame, target_var: str):
    y = None

    # Only update "y" if a target is given (i.e. if the inputted DF is for training)
    if target_var is not None:
        # Remove samples with missing prices and create prediction target for training
        X.dropna(axis=0, subset=[target_var], inplace=True)
        y = X.SalePrice
        X.drop([target_var], axis=1, inplace=True) # Drop target from training set (house prices)

    return X, y

def split_columns_by_type(X: pd.DataFrame, card_thresh: int):
    # Fetch list of numerical (ints and floats) and categorical ('object') columns 
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

    # Divide categorical columns into low and high cardinality
    low_cardinality_cols = [col for col in categorical_cols if X[col].nunique() < card_thresh]
    high_cardinality_cols = list(set(categorical_cols) - set(low_cardinality_cols))

    return numerical_cols, low_cardinality_cols, high_cardinality_cols

def encode_categorical_columns(X_train: pd.DataFrame, X_test: pd.DataFrame, low_card_cols: list, high_card_cols: list):
    # Ordinal encoding for categorical variables with high cardinality
    ## TODO: Try other ways of dealing with unknown values in test data during transform
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal_X_train = pd.DataFrame(ordinal_encoder.fit_transform(X_train[high_card_cols]))
    ordinal_X_test = pd.DataFrame(ordinal_encoder.transform(X_test[high_card_cols]))

    # One-hot encoding for low cardinality
    ## TODO: Try other ways of dealing with unknown values in test data during transform
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_X_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_card_cols]))
    OH_X_test = pd.DataFrame(OH_encoder.transform(X_test[low_card_cols]))

    # Put indices back
    OH_X_train.index = X_train.index
    OH_X_test.index = X_test.index

    return ordinal_X_train, OH_X_train, ordinal_X_test, OH_X_test

def prepare_data(X_train: pd.DataFrame, X_test: pd.DataFrame, target_var: str, card_thresh: int = 10, verbose: bool = False):
    # Split training data into input features (X) and target output (y)
    original_n_cols = len(X_train.columns)
    X_train, y_train = split_X_y(X_train, target_var)
    n_cols_after_split = len(X_train.columns)

    # Split columns by type and encode the categorical ones
    numerical_cols, low_card_cols, high_card_cols = split_columns_by_type(X_train, card_thresh)
    ordinal_X_train, OH_X_train, ordinal_X_test, OH_X_test = encode_categorical_columns(X_train, X_test, low_card_cols, high_card_cols)

    # Put together all types of data (for TRAINING data)
    ## Reset index to avoid adding samples
    X_train = pd.concat([X_train[numerical_cols].reset_index(drop=True),
                   ordinal_X_train.reset_index(drop=True),
                   OH_X_train.reset_index(drop=True)],
                   axis=1)
    X_train.columns = X_train.columns.astype(str) # Columns shall have a string type
    
    # Put together all types of data (for TESTING data)
    ## Reset index to avoid adding samples
    X_test = pd.concat([X_test[numerical_cols].reset_index(drop=True),
                   ordinal_X_test.reset_index(drop=True),
                   OH_X_test.reset_index(drop=True)],
                   axis=1)
    X_test.columns = X_test.columns.astype(str) # Columns shall have a string type

    # Perform imputation for missing values
    ## TODO: Experiment with other ways of dealing with them: Dropping or extended imputation
    my_imputer = SimpleImputer()
    X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    X_test = pd.DataFrame(my_imputer.transform(X_test))
    
    # Print information about data (if requested)
    if verbose:
        n_num_cols = len(numerical_cols)
        n_low_card_cols = len(low_card_cols)
        n_high_card_cols = len(high_card_cols)
        n_categorical_cols = n_low_card_cols + n_high_card_cols
        n_cols = n_num_cols + n_categorical_cols

        # Check for unexpected differences in the amount of available variables
        if n_cols != n_cols_after_split:
            print('WARNING: The number of columns after removing the target variable is NOT the same as the one after processing the categorical variables!')

        print(f'\nOriginally, there were {original_n_cols} variables in the training set ({n_cols_after_split} after removing the target variable). {n_num_cols} are numerical variables and {n_categorical_cols} are categorical, from which {n_low_card_cols} have low cardinality and {n_high_card_cols} have high cardinality.\n')

        print(f'Numerical variables:\n  {numerical_cols}\n')

        print(f'Categorical variables: ')
        print(f'  - Low cardinality (< {card_thresh} different values): {low_card_cols}\n')
        print(f'  - High cardinality (> {card_thresh} different values): {high_card_cols}')

    return X_train, y_train, X_test

def build_and_train_model(X: pd.DataFrame, y: pd.Series, n_trees: int = 100):
    # Build and train final model with all training set
    print(f'\nThere are {len(X)} input samples and {len(y)} output samples')
    print(f'A Random Forest Regressor with {n_trees} estimators will be used')
    final_model = RandomForestRegressor(n_estimators=n_trees, criterion='absolute_error', random_state=0)
    final_model.fit(X.values, y.values) # Train with np.NDArray, not pd.DF

    return final_model

def predict_and_generate_submission(model: RandomForestRegressor, X_test:pd.DataFrame, path_to_submission_csv: str):
    # Generate test predictions and generate submission CSV for Kaggle competition
    preds_test = model.predict(X_test.values) # Predict with np.NDArray, not pd.DF
    output = pd.DataFrame({'Id': X_test.index+1461, 'SalePrice': preds_test})
    output.to_csv(path_to_submission_csv, index=False)
    print(f'\nSubmission CSV created in {path_to_submission_csv}')

def main(path_to_train_csv: str, path_to_test_csv: str, target_var: str, card_thresh: int, path_to_submission_csv: str, n_trees: int):
    # Load Kaggle Housing Prices competition data
    X_train = pd.read_csv(path_to_train_csv, index_col='Id')
    X_test = pd.read_csv(path_to_test_csv, index_col='Id')

    # Prepare train and test data
    ## TODO: Experiment with different cardenality thresholds
    X_train, y_train, X_test = prepare_data(X_train, X_test, target_var, card_thresh=card_thresh, verbose=True)

    # Build, train, and use model
    model = build_and_train_model(X_train, y_train, n_trees)
    predict_and_generate_submission(model, X_test, path_to_submission_csv)

if __name__ == '__main__':
    main(PATH_TO_TRAIN_CSV, PATH_TO_TEST_CSV, TARGET_VAR, CARDENALITY_THRESHOLD, PATH_TO_SUBMISSION_CSV, N_TREES)
