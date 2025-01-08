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
TODO


## [Lesson 4: Pipelines](https://www.kaggle.com/code/alexisbcook/pipelines)
TODO


## [Lesson 5: Cross Validation](https://www.kaggle.com/code/alexisbcook/cross-validation)
TODO


## [Lesson 6: XGBoost](https://www.kaggle.com/code/alexisbcook/xgboost)
TODO


## [Lesson 7: Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage)
TODO
