# Intro to Machine Learning
This course is estimated to last 3 hours, builds on Python, and prepares for the following courses:
- [Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability)
- [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)
- [Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)

A `requirements.txt` file is included. It lists all the required dependencies to run the coding exercises present in this course. To install it, please execute: `pip install -r requirements.txt`

## [Lesson 1: How Models Work](https://www.kaggle.com/code/dansbecker/how-models-work)
### Introduction
Machine Learning is about identifying patterns in order to make predictions. An example modle that does this are the **decision trees**. A very basic example would be to check a certain variable (e.g., the number of bedrooms in a house) to classify the inputs (e.g., a house) into different categories (e.g., price ranges).

The process of finding out the patters in the data is called ***fitting*** or ***training***. The data used to *fit* the model is called *training data*. Once the model has been fit, you can use it with new data to do *predictions*.

### Improving the Decision Tree
You can add more layers to your decision trees to further classify the already classified inputs. For example, after checking the number of bedrooms, you could check the square meters to get a finer prediction of the market price of a house. Each layer is called a **split** and the more of them a decision tree has, the *deeper* it is. The bottom layers with the predicted price for a given house are called **leafs**.


## [Lesson 2: Bassic Data Exploration](https://www.kaggle.com/code/dansbecker/basic-data-exploration)
### Using Pandas to Get Familiar with Your Data
Pandas is a very popular Python module to explore and manipulate data. Most people will abbreviate it as `pd` while importing it. Pandas is built around the concept of a `DataFrame`, which is very similar to a *table* in an SQL databaase or to a *sheet* in Excel. To play around with it, you can use [data about the Melbourne housing prices](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot). The example exploration will be available in the [data exploration script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises/lesson02__data_exploration.py).

### Interpreting Data Description
It uses the [`DataFrame.describe()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) function to print the first 8 *elements* (rows) of each column in the `DataFrame`. The first element (`count`) shows how many rows have non-missing values (e.g., we can't have the size of the 2nd bedroom of a house with a single bedroom). The second and third elements (`mean` and `std`) are the average and the standard deviation of the column, respectively. The following values (`min`, `25%`, `50%`, `75%` and `max`) are the minimum and maximum values of the column and number bigger than the indicated % (and smaller than 100-X %). These last ones are called the *25th, 50th, and 75th percentiles*.

This lesson includes an online [coding exercise](https://www.kaggle.com/code/jmtc7kaggle/exercise-explore-your-data/edit). I added all the solutions to the [data exploration script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises/lesson02__data_exploration.py) mentioned before.


## [Lesson 3: Your First Machine Learning Model](https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model)
### Selecting Data for  Modeling
Sometimes the problem to be solved has to rely on too many variables to handle all of them manually. We could ignore some based on our experience and intuition. Additionally, there are some techniques to determine which are more relevant, such as [PCA (Principal Component Analysis)](https://en.wikipedia.org/wiki/Principal_component_analysis), that will be seen in later courses. The example data selection will be available in the [decision tree script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises/lesson03__decision_tree.py).

### Selecting the Prediction Target
You can select a single column using *dot-notation*. This will store all the contents of said column in a `Series`, which behaves like a `DataFrame` with a single column.

### Selecting the Features
The inputs of our model are called *features*. These will be the variables used to determine house prices. We can access a set of features by indexing a `DataFrame` with strings.

It's important to visualize and understand our input and output data in order to detect potential unforseen issues. The [`DataFrame.describe()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) and [`DataFrame.head()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html) methods are very useful for this.

### Building a Model
Python's **Scikit-learn** module is a very popular easy-to-use tool to create Machine Learning models. The steps to build a model are:
1. **Defining the type of model** to be built and its parameters
2. **Fitting the data**, using the selected features and the prediction target.
3. **Predict**, to get data for the next step
4. **Evaluate** the performance of the model

In the [decision tree script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises/lesson03__decision_tree.py), a decision tree model is defined, fitted, and used using Scikit-learn (called `sklearn` in the code). Note that the model is used to predict prices that already exist in the data used for fitting, which would not be a real case scenario.

It's also important to mention than in the example we used a **regression model** because the objective was to predict a price, which is a countinuous variable. There are also **classifier models**, whose outputs will only be `True` or `False`.

Note that in Algebra, matrices are usually represented by upper-case leters, while vectors are represented by lower-case ones. Since often in Machine Learning, the features are a set of vectors (hence, a matrix), and the prediction target is often a single vector, the features are commonly referred to as `X` (in upper-case), while `y` (in lower-case) is often used to represent the prediction targets or model outputs. It's a choice made for consistency with the Mathematical notation. However, some prefer to use more representative names for these variables, such as `features` and `predictions`.

This lesson includes an online [coding exercise](https://www.kaggle.com/kernels/fork/1404276).


## [Lesson 4: Model Validation](https://www.kaggle.com/code/dansbecker/model-validation)
### Mean Absolute Error (MAE)
Measuring the quality of a model is the key to iteratively improve it. In most models, we measure *predictive accuracy*, which represents how close the predictions are to the reality.

There are many metrics to evaluate the quality of a model, but here ***Mean Absolute Error (MAE)*** will be used. An error is the difference between the prediction and the reality. If the price of a house is 150k and we estimated 200k, there is a 50k error. If we take the absolute value of each of the errors of our predictions and then compute their mean, we would obtain the MAE. Another popular metric is the ***Mean Square Error (MSE)***. It is very similar to the MAE, but instead of using the absolute values of the errors, the square root of them is used.

In the [model validation script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises/lesson04__validation.py), a model is defined, fitted, and validated.

### Problem of *in-sample* Scores
When using the same data to fit (a.k.a. train) and evaluate a model, we end up with what is known as an *in-sample* score. This is an issue, since our data may show some pattern that is not present in the real world, which would allow our model to discover and exploit that pattern, achieving a very high *in-sample* score while its performance for new data would be way worse. Also, during the fitting process, the model is optimized for the data we showed, while the real world will almost always present more variability, including combinations of features that our model never saw during the fitting process. Therefore, when evaluating our model with the same data used to fit it, we will always get results that are way too optimistic.

The easiest way to avoid this is by putting aside a part of our data to be used exclusively for validation purposes and not for fitting. This way, when validating with new data, the obtained score will be more representative of the model's actual performance. However, since we are fitting the model with less data, it's likely to end up performing more poorly than if we would have used all of the data for fitting it. This is why the validation data shall ideally be insignificant with respect to the fitting data while still being variated and representative of the real world. A general approach for this split is to use 70% of the data for training and 30% for validation. The less data you have, the higher the percentage of the training set should become. When dealing with exceptionally low quantities of data, strategies such as the [***cross-validation (CV)***](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) can help. There are several types of it. Some examples are:
- **K-fold CV**: The data is divided in subsets (usually between 5 and 10 to balance accuracy and computational effort) and the model is trained in all subsets except one, which is used for validation. This is repeated for each possible combination of training/validation subsets. At the end we will get the mean and standard deviation (or variance) of the chosen validation metric.
- **Leave-one-out CV**: An extreme version of K-fold CV. Instead of splitting the data in subsets, we split it in instances. Therefore, we train with all the houses in our dataset except for one, which will be used to validate. This process is repeated leaving out every single one of the data instances, one at a time.
- **Nested CV**: It consist in a *meta-CV*. Using subsets of the same data for both training and validating can lead to more optimistic metrics. Nested CV solves it by having an iterative process to tune the model's parameter (which can use any of the abovementioned CVs) which will be followed by an additional validation (or *test*) step with data that has never been seen for neither training nor the 1st validation step.

**Scikit-learn** has **train_test_split** to help divide a dataset into training and testing subsets. It is also used in the abovementioned [model validation script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises/lesson04__validation.py). When using the [Melbourne housing prices data](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) without splitting between training and validation, we get an error of around 500, but when we do the splitting, it goes up to more than 250 000, which better represents the performance of the model. This very big difference shows that our model was ***overfitted*** to our training data, which means that we need more data to have better real-world results. Further details about this will be seen in **Lesson 5**.

As an additional note, the average home price is around 1.1 million, so having an error of 250k would mean an inaccuracy of a quarter of the price, which means that when training this model with this data, the result is not precise enough to be used in the real world.


## [Lesson 5: Underfitting and Overfitting](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting)
Since now we are able to quantify the performance of a model, we can now experiment with different models or parameters of our current one. In the [documentation of Scikit-learn for regression decision trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html), they specify all available parameters for that type of model. The most important one is the tree's depth. Generally, **decision trees don't go deeper than 10 splits**, since that would generate 2<sup>10</sup> leaves, meaning that the data is split into 1024 subsets, which already requires a lot of training data to have enough examples for each leaf to avoid ***overfitting* to the training data**.

On the other side, if having an over-simplified model with only 2 or 4 leafs, way too many houses will be used for each of them and the model won't be able to capture the complexity of the problem. This is called ***underfitting***.

When both training and validation scores are low, it's a sign of underfitting. If the training score is high but the validation one is low, the model is likely overfitted. In Machine Learning, the objective is to have as much data as possible and to choose a model that is able to represent it well enough with parameters that avoid overfitting.

### Managing the Depth of a Decision Tree
The depth of a decision tree can be managed in many ways. One of the options is to set the `max_leaf_nodes` argument. The more leaf nodes, the further away we move from underfitting and the closer we get to overfitting.

In the [overfitting and underfitting script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises/lesson05__over_and_underfitting.py), a method is defined to define, train, and evaluate multiple decision trees with 5, 50, 500, and 5000 leaves. The smallest validation error was obtained for the tree with 500 leaves.


## [Lesson 6: Random Forests](https://www.kaggle.com/code/dansbecker/random-forests)
Decision trees are a simple ML model very that is very easy to overfit or to underfit. However, there are many more modern alternatives, such as random forests. They use many trees and average their predictions to generate a single, more accurate prediction. They often offer quite good performance regardless of their parameters. Other models may be able to reach better results, but their parameters are harder to get right.

In the [random forest script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises/lesson06__random_forest.py), a random forest is defined, trained, and evaluated. It gives a MAE of just under 200k, while the best we got with decision trees was 250k.


## [Lesson 7: Machine Learning Competitions](https://www.kaggle.com/code/alexisbcook/machine-learning-competitions)
In order to improve the knowledge on ML and get practical experience, Kaggle's ML competitions are a great option, since they provide a problem and the data to solve it. An example of them is the [House Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course).

After completing this course, good next steps are:
- [Kaggle ML competitions](https://www.kaggle.com/competitions) to learn how to optimize random forest and experiment with data on our own.
- [Intermediate ML course](https://www.kaggle.com/learn/intermediate-machine-learning) to learn how to handle missing data and how to use *xgboost* to get even better results than the ones we get with a random forest.
- [Pandas course](https://www.kaggle.com/Learn/Pandas) to improve our data manipulation skills.
- [Introduction to Deep Learning course](https://www.kaggle.com/Learn/intro-to-Deep-Learning) to build advanced Computer Vision models that perform even better than humans.