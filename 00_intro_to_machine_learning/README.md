# Intro to Machine Learning
This course is estimated to last 3 hours, builds on Python, and prepares for the following courses:
- [Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability)
- [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)
- [Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)


## [Lesson 1: How Models Work](https://www.kaggle.com/code/dansbecker/how-models-work)
### Introduction
Machine Learning is about identifying patterns in order to make predictions. An example modle that does this are the **decision trees**. A very basic example would be to check a certain variable (e.g., the number of bedrooms in a house) to classify the inputs (e.g., a house) into different categories (e.g., price ranges).

The process of finding out the patters in the data is called ***fitting*** or ***training***. The data used to *fit* the model is called *training data*. Once the model has been fit, you can use it with new data to do *predictions*.

### Improving the Decision Tree
You can add more layers to your decision trees to further classify the already classified inputs. For example, after checking the number of bedrooms, you could check the square meters to get a finer prediction of the market price of a house. Each layer is called a **split** and the more of them a decision tree has, the *deeper* it is. The bottom layers with the predicted price for a given house are called **leafs**.


## [Lesson 2: Bassic Data Exploration](https://www.kaggle.com/code/dansbecker/basic-data-exploration)
### Using Pandas to Get Familiar with Your Data
Pandas is a very popular Python module to explore and manipulate data. Most people will abbreviate it as `pd` while importing it. Pandas is built around the concept of a `DataFrame`, which is very similar to a *table* in an SQL databaase or to a *sheet* in Excel. To play around with it, you can use [data about the Melbourne housing prices](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot). The example exploration will be available in the [Melbourne housing prices exploration script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises\lesson02_melbourne_housing_prices_explo.py).

### Interpreting Data Description
It uses the [`DataFrame.describe()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) function to print the first 8 *elements* (rows) of each column in the `DataFrame`. The first element (`count`) shows how many rows have non-missing values (e.g., we can't have the size of the 2nd bedroom of a house with a single bedroom). The second and third elements (`mean` and `std`) are the average and the standard deviation of the column, respectively. The following values (`min`, `25%`, `50%`, `75%` and `max`) are the minimum and maximum values of the column and number bigger than the indicated % (and smaller than 100-X %). These last ones are called the *25th, 50th, and 75th percentiles*.

This lesson includes an online [coding exercise](https://www.kaggle.com/code/jmtc7kaggle/exercise-explore-your-data/edit). I added all the solutions to the [Melbourne housing prices exploration script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises\lesson02__melbourne_housing_prices_explo.py) mentioned before.


## [Lesson 3: Your First Machine Learning Model](https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model)
### Selecting Data for  Modeling
Sometimes the problem to be solved has to rely on too many variables to handle all of them manually. We could ignore some based on our experience and intuition. Additionally, there are some techniques to determine which are more relevant, such as [PCA (Principal Component Analysis)](https://en.wikipedia.org/wiki/Principal_component_analysis), that will be seen in later courses. The example data selection will be available in the [Melbourne housing prices decision tree script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises\lesson03_melbourne_housing_prices_decision_tree.py).

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

In the [Melbourne housing prices decision tree script](https://github.com/jmtc7/kaggle-courses/tree/main/00_intro_to_machine_learning/coding_exercises\lesson03_melbourne_housing_prices_decision_tree.py), a decision tree model is defined, fitted, and used using Scikit-learn (called `sklearn` in the code). Note that the model is used to predict prices that already exist in the data used for fitting, which would not be a real case scenario.

It's also important to mention than in the example we used a **regression model** because the objective was to predict a price, which is a countinuous variable. There are also **classifier models**, whose outputs will only be `True` or `False`.

This lesson includes an online [coding exercise](https://www.kaggle.com/kernels/fork/1404276).

## [Lesson 4: Model Validation](https://www.kaggle.com/code/dansbecker/model-validation)
TODO


## [Lesson 5: Underfitting and Overfitting](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting)
TODO


## [Lesson 6: Random Forests](https://www.kaggle.com/code/dansbecker/random-forests)
TODO


## [Lesson 7: Machine Learning Competitions](https://www.kaggle.com/code/alexisbcook/machine-learning-competitions)
TODO
