# Intermediate Machine Learning
This course is estimated to last 4 hours, builds on Python, and prepares for the following courses:
- [Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- [Time Series](https://www.kaggle.com/learn/time-series)

A `requirements.txt` file is included. It lists all the required dependencies to run the coding exercises present in this course. To install it, please execute: `pip install -r requirements.txt`

## [Lesson 1: Introduction](https://www.kaggle.com/code/alexisbcook/introduction)
This course will go trough more realistic usecases than the [Introduction to Machine Learning](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning) one, with topics including:
- Handling of real world data, with missing values and categorical variables
- Designing pipelines to improve ML models
- Advanced validation methods (cross-validation)
- Build and use state-of-the-art ML models (XGBoost)
- Avoiding common ML issues (leakage)

There will be some coding exercises, which will use the data from [Kaggle's Housing Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course). The predictions will be submitted to see how we rise in the leaderboard as we improve our model.

An example script refreshing how to read data, build a random forest, train it, and generate predictions will be available in the [ML review script](https://github.com/jmtc7/kaggle-courses/tree/main/01_intermediate_machine_learning/coding_exercises/lesson01__ml_intro_review.py). It generates the `lesson01__submission.csv`, which can be used to submit predictions to the [Kaggle's Housing Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course). It gives a public score of 20998.8378.


## [Lesson 2: Missing Values](https://www.kaggle.com/code/alexisbcook/missing-values)
In data from real world datasets there are often missing values. There are several approaches to deal with it:
- **Dropping all variables with missing data** is the simplest option. However, unless most of the values are missing, this would negatively affect the model significantly, since relevant information may be represented by changes in this variable.
- **Imputation** is a method that consist in filling up the missing values. A simple approach is to use the mean value of the column. It will very likely be wrong, but it's better than dropping the entire column. More advanced guesses, such as **regression imputation** can be used, but the complexity of the solution escalates quickly and it usually doesn't end in big performance improvements, specially when dealing with advanced ML algotihms.
- **Extended imputation** consist in adding a boolean column for each of the columns with missing values. It will be set to `True` whenever an imputation is performed. This can help with some ML models, but it is absolutely useless with others.

The [missing values script](https://github.com/jmtc7/kaggle-courses/tree/main/01_intermediate_machine_learning/coding_exercises/lesson02__missing_values.py) implements each of these options and evaluates the MAE of a default random forest when using each of them. The obtained MAEs are 175703, 169237, and 169795, respectively. From the 12 used columns, 3 contain missing values, so removing them removes a significant amount of the available data. This is why imputation methods perform better than removing the columns. The difference between the results when using regular imputation vs the extended one come from the fact that, given that we added some columns, the input data is not the same, so some variance is expected. We saw, however, that the model did NOT benefit from the additional information.

It's worth to mention that sometimes **imputation may worsen the results** due to noisy data or the wrong imputation strategy being used. For example, when filling up missing data about the year in which the house's garage was built by using the mean, we may be adding a very misleading number, since the sample may be missing that data because they don't have a garage at all.

The [missing values script](https://github.com/jmtc7/kaggle-courses/tree/main/01_intermediate_machine_learning/coding_exercises/lesson02__missing_values.py) generates the `lesson02__submission.csv`, which can be used to submit predictions to the [Kaggle's Housing Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course). It gives a public score of 16719.3727.


## [Lesson 3: Categorical Variables](https://www.kaggle.com/code/alexisbcook/categorical-variables)
Variables that can only have a finite amount of values are known as ***categorical*** or ***discrete*** variables. An example would be a field in a `DataFrame` that contains the brand of a vehicle. There are only a few possible valid values for such a field. To use them in ML algorithms, they often need some preprocessing. These are the most common approaches:
- **Dropping categorical variables**: The simplest way to deal with them is not using them. However, they may hide relevant information that we won't be able to take advantage of.
- **Ordinal encoding**: It assigns each possible value an integer value. This assumes that the categorical variable can be sorted in some way. For example: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3). This approach usually works well with tree-based algorithms. A common alternative is to set random integers to each possible value, but it often gives worse results than custom labels.
- **One-hot encoding**: It generates a new column for each possible value of the categorical variable. The values of these new columns will be set to `True` whenever the sample had that value in the original variable. For example, if having a `color` variable that can be `red`, `yellow`, or `green`, they can't be sorted. Therefore, we could create three new columns: `red`, `yellow`, and `green`. Lastly, we would set to `True` the column `red` and to `False` the colmuns `yellow` and `green` for all the samples whose `color` was `red`, etc. This approach's performance often drops when many values are possible, from 15 or so.

The categorical variables with an intrinsic rnaking (i.e., those which can be sorted) are known as ***ordinal variables***. The categorical variables that can't be sorted are known as ***nominal variables***. When a categorical variable can have a lot of different values, they are said to have ***high cardinality***. If it can only have a few, it has *low cardinality*. Usually, ordinal variables are dealt with by using ordinal encoding. Nominal variables with low cardinality are managed trough one-hot encoding, and nominal variables with high cardinality are usually either dropped or ordinally encoded. This is done to avoid exponential increases in the dataset size.

The [categorical variables script](https://github.com/jmtc7/kaggle-courses/tree/main/01_intermediate_machine_learning/coding_exercises/lesson03__categorical_variables.py) implements each of these options and evaluates the MAE of a random forest with 100 estimators when using each of them. For ordinal encoding, random integers are assigned to the possible values. The obtained MAEs are 175703, 165936, and 166089, respectively. We see how dropping all categorical values had a negative effect in the results, which means that they contain some relevant information. The other two approaches gave us quite similar result, so there isn't a noticeable difference between them. Usually, one-hot encoding gives the best results.

Note that when dealing with categorical data, we may face a situation in which there are some values in some of the sets of data (training, validation, or test) that don't appear in the other sets. This is problematic because it can mess up our encodings. These columns can be easily found using Python's `Sets`, as done in the [categorical variables script](https://github.com/jmtc7/kaggle-courses/tree/main/01_intermediate_machine_learning/coding_exercises/lesson03__categorical_variables.py). To solve this, you can:
- Drop the problematic categorical columns.
- Write a custom ordinal encoding that deals with unseen values by, for example, adding a new integer.

I created a dedicated [competition script](https://github.com/jmtc7/kaggle-courses/tree/main/01_intermediate_machine_learning/coding_exercises/house_prices_competition.py), in which from now on I will start to add the learnt topics to generate a submission for the competition. For this submission (commit [1fcfa0aec297ecd6de43a2f4420755a9bfdf431d](https://github.com/jmtc7/kaggle-courses/commit/1fcfa0aec297ecd6de43a2f4420755a9bfdf431d)), I used a random forest with 100 estimators and *absolute error* as the criterion. Random state set to 0. I used all numerical features in the data except the price, one-hot encoded the low-cardinality ones, and ordinal-encoded the high cardinality ones. The cardinality threshold was set to 10. I used the SimpleImputer to remove missing values. It got a public score of 16079.55172, so reduced the previous error by around 700$.


## [Lesson 4: Pipelines](https://www.kaggle.com/code/alexisbcook/pipelines)
TODO


## [Lesson 5: Cross Validation](https://www.kaggle.com/code/alexisbcook/cross-validation)
TODO


## [Lesson 6: XGBoost](https://www.kaggle.com/code/alexisbcook/xgboost)
TODO


## [Lesson 7: Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage)
TODO
