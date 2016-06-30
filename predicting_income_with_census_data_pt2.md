# Predicting Income, Part 2: Building a Classifier from Census Data
**An end-to-end machine learning example using Pandas and Scikit-Learn**    
_by Benjamin Bengfort and Rebecca Bilbro, adapted from a [post](http://blog.districtdatalabs.com/building-a-classifier-from-census-data) originally written for the [District Data Labs blog](http://blog.districtdatalabs.com/)_      

Welcome back! If you haven't read Part 1 yet, you can do so [here](https://github.com/CommerceDataService/tutorial-predicting-income/blob/master/predicting_income_with_census_data_pt1.md).

In Part 1 of this 2-part post, we discussed getting started with machine learning and conducting feature analysis and exploration using a combination of statistical and visual techniques. Recall that our objective is to use U.S. Census Bureau demographic and income data to power a command-line application that can guess whether or not the user makes more or less than $50K based on their demographic profile. Now that we've completed some initial investigation and have started to identify the information encoded in our dataset, in Part 2 we'll discuss how to transform that data and put it into a machine learning pipeline that can support our application.

Building a pipeline for machine learning enables us to manage our data flow so that the predictive model we construct can be used in a data product as an engine to create useful new data. Below is a good framework to keep in mind for how our data will flow into and out of the pipeline:

![ML Model Pipeline](figures/ml_pipeline.png)

## Data Management

Our next step is to structure our data on disk in a way that can be loaded into Scikit-Learn in a repeatable fashion for continued analysis. We suggest using the `sklearn.datasets.base.Bunch` object to load the data into `data` and `target` attributes respectively, similar to how Scikit-Learn's toy datasets are structured. Using this object to manage our data will mirror the native Scikit-Learn API and allow us to easily adapt template code from Scikit-Learn. Importantly, this API will also allow us to communicate to other developers and our future-selves about exactly how to use the data.

In order to organize our data on disk, we'll need to add the following files:

- `README.md`: a markdown file containing information about the dataset and attribution. Will be exposed by the `DESCR` attribute.    
- `meta.json`: a helper file that contains machine readable information about the dataset like `target_names` and `feature_names`.    

Using a text editor, we constructed a pretty simple `README.md` in Markdown that gave the title of the dataset, the link to the UCI Machine Learning Repository page that contained the dataset, as well as a citation to the author.

The `meta.json` file, however, we can write using the dataframe that we created in Part 1. We've already done the manual work of writing the column names into a `names` variable earlier, there's no point in letting that go to waste!


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

This code creates a `meta.json` file by inspecting the dataframe that we have constructed. The `target_names` column is just the two unique values in the `data.income` series; by using the `pd.Series.unique` method, we're guaranteed to spot data errors if there are more or less than the expected two values. The `feature_names` are simply the names of all the columns.

Then we got tricky &mdash; we want to store the possible values of each categorical field for lookup later, but how do we know which columns are categorical and which are not? Luckily, Pandas has already done an analysis for us, and has stored the column data type, `data[column].dtype`, as either `int64` or `object`. Here we are using a dictionary comprehension to create a dictionary whose keys are the categorical columns, determined by checking the object type and comparing with `object`, and whose values are a list of unique values for that field.

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

The primary work of the `load_data` function is to locate the appropriate files on disk, given a root directory that's passed in as an argument (if you saved your data in a different directory, you can modify the root to have it look in the right place). The metadata is included with the bunch, and is also used to split the train and test datasets into `data` and `target` variables appropriately, so that we'll be able to easily pass them to the Scikit-Learn `fit` and `predict` estimator methods.

## Feature Extraction

Now that our data is structured a bit more natively to Scikit-Learn, we can start to use our data to fit models. Unfortunately, the non-numeric categorical values are not useful for machine learning; we need a single instance table that contains _numeric values_. In order to extract this from the dataset, we'll have to use Scikit-Learn transformers to transform our input dataset into something that can be fit to a model. In particular, we'll have to do the following:

- encode the categorical labels as numeric data    
- impute missing values with data (or remove them)    

First we'll explore how to apply these transformations to our dataset, then we'll create a feature extraction pipeline that we can use to build a model from the raw input data. This pipeline will apply both the imputer and the label encoders directly in front of our classifier, so that we can ensure that features are extracted systematically in both the training and test datasets.  

### Label Encoding

Our first step is to get our data out of the object datatype and into a numeric type, since nearly all operations we'd like to apply to our data are going to rely on numeric types. Luckily, Scikit-Learn does provide a transformer for converting categorical labels into numeric integers: [`sklearn.preprocessing.LabelEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). Unfortunately, it can only transform a single vector at a time, so we'll have to adapt it to apply it to multiple columns.

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

In order to create a multicolumn LabelEncoder, we'll have to extend the `TransformerMixin` in Scikit-Learn to create a transformer class of our own, then provide `fit` and `transform` methods that wrap individual `LabelEncoders` for our columns. Our code, inspired by the StackOverflow post &ldquo;[Label encoding across multiple columns in scikit-learn](http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn)&rdquo;, is as follows:


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

This specialized transformer now has the ability to encode multiple column labels in a data frame, saving information about the state of the encoders. It would be trivial to add an `inverse_transform` method as well that accepts numeric data and converts it back to labels, using the `inverse_transform` method of each individual `LabelEncoder` on a per-column basis.

### Imputation

According to the `adult.names` file, unknown values are given via the `"?"` string. We'll have to either ignore rows that contain a `"?"` or impute their value to the row. Scikit-Learn provides a transformer for dealing with missing values at either the column level or at the row level in the `sklearn.preprocessing` library called the [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html).

The `Imputer` requires information about what the missing values are, either an integer or the string, `Nan` for `np.nan` data types, it then requires a strategy for dealing with it. For example, the `Imputer` can fill in the missing values with the mean, median, or most frequent values for each column. If provided an axis argument of 0, columns that contain only missing data are discarded; if provided an axis argument of 1, rows which contain only missing values raise an exception. Basic usage of the `Imputer` is as follows:

```python
imputer = Imputer(missing_values='Nan', strategy='most_frequent')
imputer.fit(dataset.data)
```

Unfortunately, this would not work for our label encoded data, because 0 can be an acceptable label. Unless we could guarantee that a 0 always started out as a `"?"`, this encoding would break our numeric columns that already had zeros in them to begin with. This is certainly a challenging problem, and unfortunately the best we can do is to once again create a custom Imputer.


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

Our custom imputer, like the `EncodeCategorical` transformer takes a set of columns to perform imputation on. In this case we only wrap a single `Imputer` as the `Imputer` is already multicolumn &mdash; all that's required is to ensure that the correct columns are transformed. We inspected the encoders and found only three columns that had missing values in them, and passed them directly into the custom imputer.

We chose to do the label encoding first, assuming that because the `Imputer` required numeric values, we'd be able to do the parsing in advance. However, after requiring a custom imputer, we'd say that it's probably best to deal with the missing values early, when they're still a specific value, rather than take a chance.

## Model Build

Now that we've finally achieved our feature extraction, we can continue on to the model build phase. To create our classifier, we're going to create a [`Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) that uses our feature transformers and ends in an estimator that can do classification. We can then write the entire pipeline object to disk with the `pickle`, allowing us to load it up and use it to make predictions in the future.

A pipeline is a step-by-step set of transformers that takes input data and transforms it, until finally passing it to an estimator at the end. Pipelines can be constructed using a named declarative syntax so that they're easy to modify and develop. Our pipeline is as follows:

```python
# Ee need to encode our target data as well
yencode = LabelEncoder().fit(dataset.target)

# Construct the pipeline
census = Pipeline([
        ('encoder',  EncodeCategorical(dataset.categorical_features.keys())),
        ('imputer', ImputeCategorical(['workclass', 'native-country', 'occupation'])),
        ('classifier', LogisticRegression())
    ])

# Fit the pipeline
census.fit(dataset.data, yencode.transform(dataset.target))
```

The pipeline first passes data through our encoder, then to the imputer, and finally to our classifier. In this case, we have chosen a `LogisticRegression`, a regularized linear model that is used to estimate a categorical dependent variable, much like the binary target we have in this case. We can then evaluate the model on the test data set using the same exact pipeline.


```python
# encode test targets
y_true = yencode.transform([y for y in dataset.target_test])

# use the model to get the predicted value
y_pred = census.predict(dataset.data_test)
```

How accurate is our classifier? We can use the built-in Scikit-Learn function `classification_report` to evaluate the predictive power of a model. A classification report provides three different evaluation metrics: precision, recall, and F1 score. Below we've provided a custom visualization tool that takes as input the Scikit-Learn classification report and produces a color-coded heatmap that will help guide our eye towards our model's successes (the darkest reds) and weaknesses (the lightest yellows):

```python
import numpy as np
from matplotlib import cm

def plot_classification_report(cr, title=None, cmap=cm.YlOrRd):
    title = title or 'Classification report'
    lines = cr.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines)-3)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            ax.text(column,row,matrix[row][column],va='center',ha='center')

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()

cr = classification_report(y_true, y_pred, target_names=dataset.target_names)
print cr
plot_classification_report(cr)
```

![Classification Report](figures/classification_report.png)

As we can see, the classifier we built does a fair job. With an overall F1 score of 0.77, it's nothing to sneer at. However, we can see from our heatmap that our model is much better at identifying people with annual incomes of less than $50K than it does those with incomes above that threshold. It is possible that an SVM, a Naive Bayes, or a k-Nearest Neighbor model would do better. It is easy to construct new models using the pipeline approach that we prepared before, and we would encourage you to try it out! Furthermore, a grid search, additional feature analysis, or domain expertise in Census data may lead to a higher scoring model than the one we quickly put together. Luckily, now that we've sorted out all the pipeline issues, we can get to work on inspecting and improving the model! You can check out [this post](https://districtdatalabs.silvrback.com/visual-diagnostics-for-more-informed-machine-learning-part-3)  to learn more about model evaluation and optimization through hyperparameter tuning.

For now, the last step is to save our model to disk for reuse later, with the `pickle` module:

```python
def dump_model(model, path='data', name='classifier.pickle'):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(model, f)

dump_model(census)
```

Note: It would be a good idea to also dump meta information about the date and time your model was built, who built the model, etc.!

## Model Operation

Now it's time to explore how to operationalize the model. To do this, we'll create a simple function that gathers input from the user on the command line, and returns a guess about their income level using the classifier model. Moreover, this function will load the pickled model into memory to ensure the latest and greatest saved model is what's being used.


```python
def load_model(path='data/classifier.pickle'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict(model, meta=meta):
    data = {} # Store the input from the user

    for column in meta['feature_names'][:-1]:
        # We cheat and use the mean value for the weighting category, figuring
        # that most users won't know what theirs is.
        if column == 'fnlwgt':
            data[column] = 189778
        else:
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

The hardest (and often most important) part about operationalizing the model is collecting user input. We want the user to be able to tell us when we guessed right and when we guessed wrong, and we want to log that new data so that when we re-run the model, we train on the additional data. Obviously in a bigger application user input could be handled with forms, automatic data gathering, and other advanced techniques. For now, hopefully this is enough to highlight how you might use a model in practice to make predictions or estimations for unknown data.

## Conclusion

Thanks for joining us for this end-to-end tutorial of a machine learning application in Python. We tried to stay true to our workflow so that you can generalize from this tutorial to do your own classification or regression-based applications with your own datasets.

As for the dataset we used for this tutorial, it's of course important to point out that income distribution in the United States is a complex issue. Income inequality is real, and in many historical (and current) datasets, factors like race, gender, and background tend to be correlated with income level. When drawing conclusions or building machine learning applications based on data about people, it's important to consider the ethical implications. Moreover, the [Census dataset used for this tutorial](https://archive.ics.uci.edu/ml/datasets/Adult) is now twenty years old. To learn more about current datasets that concern the important issue of income inequality, we recommend checking out the [MIDAAS Project](https://midaas.commerce.gov/): a new Department of Commerce website that provides a public API and developer toolkit for income data from the Census Bureau. For readers who are interested in using this tutorial as a jumping off point to build a robust application that addresses income inequality, MIDAAS is a great resource that can help you integrate the latest government data into your projects.

Now that you've had a chance to look at our walkthrough, we hope you'll try a few variations on your own and be well on your way to [operationalizing machine learning](http://pycon.districtdatalabs.com/posters/machine-learning/horizontal/ddl-machine-learning-print.png) for data science!  If you liked this post, we hope you'll check out the [District Data Labs blog](http://blog.districtdatalabs.com/) for more walkthroughs and tutorials on a range of data science topics including:     
 - [data products](http://blog.districtdatalabs.com/the-age-of-the-data-product)    
 - [data wrangling](http://blog.districtdatalabs.com/simple-csv-data-wrangling-with-python)    
 - [machine learning basics](http://blog.districtdatalabs.com/an-introduction-to-machine-learning-with-python)       
 - [natural language processing](https://districtdatalabs.silvrback.com/pycon-tutorial-nlp)    
 - [sentiment analysis](http://blog.districtdatalabs.com/modern-methods-for-sentiment-analysis)    

... and much more!
