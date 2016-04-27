# Predicting Income: Building a Classifier from Census Data
**An end-to-end machine learning example using Pandas and Scikit-Learn**    
_by Benjamin Bengfort and Rebecca Bilbro, adapted from a post originally posted on the [District Data Labs blog](http://blog.districtdatalabs.com/)_      

One of the first steps for many of those to getting into data science is learning how to build simple machine learning models using an open data set. For those who are interested in experimenting with building classification, regression, or clustering models, the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.html) is a great resources for open datasets, which have already been categorized by the machine learning tasks most appropriate for that dataset.

This tutorial will provide an end-to-end example of how to do just that using the Python programming language. We'll start by ingesting data from the UCI website, performing some initial exploratory analyses to get a sense for what's in the data, structure the data to fit a Scikit-Learn model and evaluate the results. Although the UCI repository does give advice as to what types of machine learning might be applied, we'll see through the tutorial that there is still much data wrangling and clever programming needed in order to create an effective classifier.

For those new to machine learning or to Scikit-Learn, we hope this is a practical example that may shed light on many challenges that crop up when developing predictive models. For more experienced readers, we hope that we can challenge you to try this workshop and refine the classifier with additional hyperparameter tuning!

## Getting Started

### Step 1: Preliminaries
**Libraries and Utilities**    
We’ll be using the following tools in the tutorial

```python
import os
import json
import pickle
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.datasets.base import Bunch
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
```

### Step 2: Obtaining the data
The next step is to use the UCI Machine Learning Repository to find a non-trivial dataset with which to build a model. While the example datasets included with Scikit-Learn are good examples of how to fit models, they do tend to be either trivial or overused. It's a bit more of a challenge to conduct a predictive exercise with a novel dataset that has several (more than 10) features and many instances (more than 10,000). There are around 350 datasets in the repository, categorized by things like task, attribute type, data type, area, or number of attributes or instances. We selected a [Census Income](http://archive.ics.uci.edu/ml/datasets/Census+Income) dataset that had 14 attributes and 48,842 instances. The task is to build a binary classifier that can determine from Census information whether or not a person makes more than $50k per year.

Every dataset in the repository comes with a link to the data folder, which you can click on and download directly to your computer. However, in an effort to make it easier to follow along, we've also included a simple `download_data` function that uses `requests.py` to fetch the data.

```python
CENSUS_DATASET = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
)

def download_data(path='data', urls=CENSUS_DATASET):
    if not os.path.exists(path):
        os.mkdir(path)

    for url in urls:
        response = requests.get(url)
        name = os.path.basename(url)
        with open(os.path.join(path, name), 'w') as f:
            f.write(response.content)

download_data()
```

This code also helps us start to think about how we're going to manage our data on disk. We've created a `data` folder in our current working directory to hold the data as it's downloaded. In the data management section, we'll expand this folder to be loaded as a `Bunch` object. `Bunches` are objects native to Scikit-Learn and are merely a simple holder with fields that can be both accessed as Python `dict` keys or object attributes for convenience (for example, "target_names" will hold the list of the names of all the labels).

## Data Exploration
The next thing to do is to explore the dataset and see what's inside. The three files that we downloaded do not have a file extension, but they are simply text files. You can change the extension to `.txt` for easier exploration if that helps. By using the `head` and `wc -l` commands on the command line, our files appear to be as follows:

- `adult.data`: A CSV dataset containing 32,562 rows and no header
- `adult.names`: A text file containing meta information about the dataset
- `adult.test`: A CSV dataset containing 16,283 rows with a weird first line

Clearly this dataset is intended to be used for machine learning, and a test and training data set has already been constructed. Similar types of split datasets are used for [Kaggle](https://www.kaggle.com/) competitions and academic conferences. This will save us a step when it comes to evaluation time.

Since we already have a CSV file, let's explore the dataset using Pandas. Because the CSV data doesn't have a header row, we will have to supply the names directly to the `pd.read_csv` function. To get these names, we manually transcribed the list from the `adult.names` file. In the future, we'll store these names as a machine readable JSON file so that we don't have to construct the list manually each time.    

```python
names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

data = pd.read_csv('data/adult.data', sep="\s*,", names=names)
print data.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>


We can see by glancing at the first 5 rows of the data that we have primarily categorical data. Our target, `data.income` is also currently constructed as a categorical field. Unfortunately, with categorical fields, we don't have a lot of visualization options (quite yet). However, it would be interesting to see the frequencies of each class, relative to the target of our classifier. To do this, we can use the `countplot` function from the Python visualization package Seaborn to count the occurrences of each data point. Let's take a look at the counts of `data.occupation` and `data.education` &mdash; two likely predictors of income in the Census data:

```python
sns.countplot(y='occupation', hue='income', data=data,)
sns.plt.show()
```

![png](census_files/census_6_1.png)


```python
sns.countplot(y='education', hue='income', data=data,)
sns.plt.show()
```

![png](census_files/census_7_1.png)


The `countplot` function accepts either an `x` or a `y` argument to specify if this is a bar plot or a column plot. We chose to use the `y` argument so that the labels were readable. The `hue` argument specifies a column for comparison; in this case we're concerned with the relationship of our categorical variables to the target income. Go ahead and explore other variables in the dataset, for example `data.race` and `data.sex` to see if those values are predictive of the level of income or not!

How do years of education correlate to income, disaggregated by race? More education does not result in the same gains in income for Asian Americans/Pacific Islanders and Native Americans compared to Caucasians:    
```python
g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
g = g.map(sns.boxplot, 'income', 'education-num')
sns.plt.show()
```
![Education and Income by Race](census_files/ed_inc_race.png)

How do years of education correlate to income, disaggregated by sex? More education also does not result in the same gains in income for women compared to men:    
```python
g = sns.FacetGrid(data, col='sex', size=4, aspect=.5)
g = g.map(sns.boxplot, 'income', 'education-num')
sns.plt.show()
```
![Education and Income by Sex](census_files/ed_inc_sex.png)

How does age correlates to income, disaggregated by race? Generally older people make more, except for Asian Americans/Pacific Islanders:    
```python
g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
g = g.map(sns.boxplot, 'income', 'age')
sns.plt.show()
```
![Age and Income by Race](census_files/age_inc_race.png)

How do hours worked per week correlates to income, disaggregated by marital status?
```python
g = sns.FacetGrid(data, col='marital-status', size=4, aspect=.5)
g = g.map(sns.boxplot, 'income', 'hours-per-week')
sns.plt.show()
```
![Hours and Income by Marital Status](census_files/hours_inc_marital.png)


## Data Management

Now that we've completed some initial investigation and have started to identify the possible features available in our dataset, we need to structure our data on disk in a way that can be loaded into Scikit-Learn in a repeatable fashion for continued analysis. We suggest using the `sklearn.datasets.base.Bunch` object to load the data into `data` and `target` attributes respectively, similar to how Scikit-Learn's toy datasets are structured. Using this object to manage our data will mirror the native Scikit-Learn API and allow us to easily copy-and-paste code that demonstrates classifiers and techniques with the built-in datasets. Importantly, this API will also allow us to communicate to other developers and our future-selves about exactly how to use the data.

In order to organize our data on disk, we'll need to add the following files:

- `README.md`: a markdown file containing information about the dataset and attribution. Will be exposed by the `DESCR` attribute.
- `meta.json`: a helper file that contains machine readable information about the dataset like `target_names` and `feature_names`.

Using a text editor, we constructed a pretty simple `README.md` in Markdown that gave the title of the dataset, the link to the UCI Machine Learning Repository page that contained the dataset, as well as a citation to the author.

The `meta.json` file, however, we can write using the data frame that we already have. We've already done the manual work of writing the column names into a `names` variable earlier, there's no point in letting that go to waste!


```python
meta = {
    'target_names': list(data.income.unique()),
    'feature_names': list(data.columns),
    'categorical_features': {
        column: list(data[column].unique())
        for column in data.columns
        if data[column].dtype == 'object'
    },
}

with open('data/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
```

This code creates a `meta.json` file by inspecting the data frame that we have constructed. The `target_names` column is just the two unique values in the `data.income` series; by using the `pd.Series.unique` method - we're guaranteed to spot data errors if there are more or less than the expected two values. The `feature_names` is simply the names of all the columns.

Then we get tricky &mdash; we want to store the possible values of each categorical field for lookup later, but how do we know which columns are categorical and which are not? Luckily, Pandas has already done an analysis for us, and has stored the column data type, `data[column].dtype`, as either `int64` or `object`. Here we are using a dictionary comprehension to create a dictionary whose keys are the categorical columns, determined by checking the object type and comparing with `object`, and whose values are a list of unique values for that field.

Now that we have everything we need stored on disk, we can create a `load_data` function, which will allow us to load the training and test datasets appropriately from disk and store them in a `Bunch`:


```python
def load_data(root='data'):
    # Load the meta data from the file
    with open(os.path.join(root, 'meta.json'), 'r') as f:
        meta = json.load(f)

    names = meta['feature_names']

    # Load the readme information
    with open(os.path.join(root, 'README.md'), 'r') as f:
        readme = f.read()

    # Load the training and test data, skipping the bad row in the test data
    train = pd.read_csv(os.path.join(root, 'adult.data'), names=names)
    test  = pd.read_csv(os.path.join(root, 'adult.test'), names=names, skiprows=1)

    # Remove the target from the categorical features
    meta['categorical_features'].pop('income')

    # Return the bunch with the appropriate data chunked apart
    return Bunch(
        data = train[names[:-1]],
        target = train[names[-1]],
        data_test = test[names[:-1]],
        target_test = test[names[-1]],
        target_names = meta['target_names'],
        feature_names = meta['feature_names'],
        categorical_features = meta['categorical_features'],
        DESCR = readme,
    )

dataset = load_data()
```

The primary work of the `load_data` function is to locate the appropriate files on disk, given a root directory that's passed in as an argument (if you saved your data in a different directory, you can modify the root to have it look in the right place). The meta data is included with the bunch, and is also used split the train and test datasets into `data` and `target` variables appropriately, such that we can pass them correctly to the Scikit-Learn `fit` and `predict` estimator methods.

## Feature Extraction

Now that our data is structured a bit more natively to Scikit-Learn, we can start to use our data to fit models. Unfortunately, the non-numeric categorical values are not useful for machine learning; we need a single instance table that contains _numeric values_. In order to extract this from the dataset, we'll have to use Scikit-Learn transformers to transform our input dataset into something that can be fit to a model. In particular, we'll have to do the following:

- encode the categorical labels as numeric data
- impute missing values with data (or remove them)

We will explore how to apply these transformations to our dataset, then we will create a feature extraction pipeline that we can use to build a model from the raw input data. This pipeline will apply both the imputer and the label encoders directly in front of our classifier, so that we can ensure that features are extracted appropriately in both the training and test datasets.  

### Label Encoding

Our first step is to get our data out of the object datatype and into a numeric type, since nearly all operations we'd like to apply to our data are going to rely on numeric types. Luckily, Sckit-Learn does provide a transformer for converting categorical labels into numeric integers: [`sklearn.preprocessing.LabelEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). Unfortunately, it can only transform a single vector at a time, so we'll have to adapt it in order to apply it to multiple columns.

Like all Scikit-Learn transformers, the `LabelEncoder` has `fit` and `transform` methods (as well as a special all-in-one, `fit_transform` method) that can be used for stateful transformation of a dataset. In the case of the `LabelEncoder`, the `fit` method discovers all unique elements in the given vector, orders them lexicographically, and assigns them an integer value. These values are actually the indices of the elements inside the `LabelEncoder.classes_` attribute, which can also be used to do a reverse lookup of the class name from the integer value.

For example, if we were to encode the `gender` column of our dataset as follows:

```python
gender = LabelEncoder()
gender.fit(dataset.data.sex)
print gender.classes_
```

We can then transform a single vector into a numeric vector as follows:

```python
print gender.transform([
    'Female', 'Female', 'Male', 'Female', 'Male'
])
```

Obviously this is very useful for a single column, and in fact the `LabelEncoder` really was intended to encode the target variable, not necessarily categorical data expected by the classifiers.

In order to create a multicolumn LabelEncoder, we'll have to extend the `TransformerMixin` in Scikit-Learn to create a transformer class of our own, then provide `fit` and `transform` methods that wrap individual `LabelEncoders` for our columns. My code, inspired by the StackOverflow post &ldquo;[Label encoding across multiple columns in scikit-learn](http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn)&rdquo;, is as follows:


```python
class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """

    def __init__(self, columns=None):
        self.columns  = columns
        self.encoders = None

    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns

        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns
        }
        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])

        return output

encoder = EncodeCategorical(dataset.categorical_features.keys())
data = encoder.fit_transform(dataset.data)
```

This specialized transformer now has the ability to encode multiple column labels in a data frame, saving information about the state of the encoders. It would be trivial to add an `inverse_transform` method as well that accepts numeric data and converts it to labels, using the `inverse_transform` method of each individual `LabelEncoder` on a per-column basis.

### Imputation

According to the `adult.names` file, unknown values are given via the `"?"` string. We'll have to either ignore rows that contain a `"?"` or impute their value to the row. Scikit-Learn provides a transformer for dealing with missing values at either the column level or at the row level in the `sklearn.preprocessing` library called the [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html).

The `Imputer` requires information about what missing values are, either an integer or the string, `Nan` for `np.nan` data types, it then requires a strategy for dealing with it. For example, the `Imputer` can fill in the missing values with the mean, median, or most frequent values for each column. If provided an axis argument of 0 then columns that contain only missing data are discarded; if provided an axis argument of 1, then rows which contain only missing values raise an exception. Basic usage of the `Imputer` is as follows:

```python
imputer = Imputer(missing_values='Nan', strategy='most_frequent')
imputer.fit(dataset.data)
```

Unfortunately, this would not work for our label encoded data, because 0 is an acceptable label &mdash; unless we could guarantee that 0 was always `"?"`, then this would break our numeric columns that already had zeros in them. This is certainly a challenging problem, and unfortunately the best we can do is to once again create a custom Imputer.


```python
class ImputeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.imputer = None

    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to impute.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns

        # Fit an imputer for each column in the data frame
        self.imputer = Imputer(missing_values=0, strategy='most_frequent')
        self.imputer.fit(data[self.columns])

        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        output[self.columns] = self.imputer.transform(output[self.columns])

        return output


imputer = ImputeCategorical(['workclass', 'native-country', 'occupation'])
data = imputer.fit_transform(data)
```

Our custom imputer, like the `EncodeCategorical` transformer takes a set of columns to perform imputation on. In this case we only wrap a single `Imputer` as the `Imputer` is already multicolumn &mdash; all that's required is to ensure that the correct columns are transformed. We inspected the encoders and found only three columns that had missing values in them, and passed them directly into the customer imputer.

We chose to do the label encoding first, assuming that because the `Imputer` required numeric values, we'd be able to do the parsing in advance. However, after requiring a custom imputer, we'd say that it's probably best to deal with the missing values early, when they're still a specific value, rather than take a chance.

## Model Build

Now that we've finally achieved our feature extraction, we can continue on to the model build phase. To create our classifier, we're going to create a [`Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) that uses our feature transformers and ends in an estimator that can do classification. We can then write the entire pipeline object to disk with the `pickle`, allowing us to load it up and use it to make predictions in the future.

A pipeline is a step-by-step set of transformers that takes input data and transforms it, until finally passing it to an estimator at the end. Pipelines can be constructed using a named declarative syntax so that they're easy to modify and develop. Our pipeline is as follows:

```python
# we need to encode our target data as well.
yencode = LabelEncoder().fit(dataset.target)

# construct the pipeline
census = Pipeline([
        ('encoder',  EncodeCategorical(dataset.categorical_features.keys())),
        ('imputer', ImputeCategorical(['workclass', 'native-country', 'occupation'])),
        ('classifier', LogisticRegression())
    ])

# fit the pipeline
census.fit(dataset.data, yencode.transform(dataset.target))
```

The pipeline first passes data through our encoder, then to the imputer, and finally to our classifier. In this case, we have chosen a `LogisticRegression`, a regularized linear model that is used to estimate a categorical dependent variable, much like the binary target we have in this case. We can then evaluate the model on the test data set using the same exact pipeline.


```python
# encode test targets, and strip trailing '.'
y_true = yencode.transform([y.rstrip(".") for y in dataset.target_test])

# use the model to get the predicted value
y_pred = census.predict(dataset.data_test)

# execute classification report
print classification_report(y_true, y_pred, target_names=dataset.target_names)
```

As part of the process in encoding the target for the test data, we discovered that the classes in the test data set had a `"."` appended to the end of the class name, which we had to strip in order for the encoder to work! However, once done, we could predict the y values using the test dataset, passing the predicted and true values to the classifier report.

The classifier we built does an ok job, with an F1 score of 0.77, nothing to sneer at. However, it is possible that an SVM, a Naive Bayes, or a k-Nearest Neighbor model would do better. It is easy to construct new models using the pipeline approach that we prepared before, and we would encourage you to try it out! Furthermore, a grid search or feature analysis may lead to a higher scoring model than the one we quickly put together. Luckily, now that we've sorted out all the pipeline issues, we can get to work on inspecting and improving the model!

The last step is to save our model to disk for reuse later, with the `pickle` module:

```python
def dump_model(model, path='data', name='classifier.pickle'):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(model, f)

dump_model(census)
```

Note: It would be a good idea to also dump meta information about the date and time your model was built, who built the model, etc.!

## Model Operation

Now it's time to explore how to use the model. To do this, we'll create a simple function that gathers input from the user on the command line, and returns a prediction with the classifier model. Moreover, this function will load the pickled model into memory to ensure the latest and greatest saved model is what's being used.


```python
def load_model(path='data/classifier.pickle'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict(model, meta=meta):
    data = {} # Store the input from the user

    for column in meta['feature_names'][:-1]:
        # Get the valid responses
        valid = meta['categorical_features'].get(column)

        # Prompt the user for an answer until good
        while True:
            val = " " + raw_input("enter {} >".format(column))
            if valid and val not in valid:
                print "Not valid, choose one of {}".format(valid)
            else:
                data[column] = val
                break

    # Create prediction and label
    yhat = model.predict(pd.DataFrame([data]))
    print "We predict that you make %s" % yencode.inverse_transform(yhat)[0]


# Execute the interface
model = load_model()
predict(model)
```

The hardest part about operationalizing the model is collecting user input. Obviously in a bigger application this could be handled with forms, automatic data gathering, and other advanced techniques. For now, hopefully this is enough to highlight how you might use the model in practice to make predictions on unknown data.

## Conclusion

This walkthrough was an end-to-end look at how we performed a classification analysis of a Census dataset that we downloaded from the Internet. We tried to stay true to our workflow so that you could get a sense for how we had to go about doing things with little to no advanced knowledge. As a result, there are definitely some things we might change if we were going to do this over. For example, given another chance we would definitely wrangle and clean both datasets and save them back to disk. Even just little things like the "." at the end of the class names in the test set were annoyances that could have been easily dealt with. But now that you've had a chance to look at our walkthrough, try a few variations on your own and you'll be well on your way to operationalizing machine learning for data science!
