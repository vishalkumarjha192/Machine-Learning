# Machine Learning


## Table of Contents

1.  [What is Machine Learning?](#1-what-is-machine-learning)
2.  [Traditional Programming vs Machine
    Learning](#2-traditional-programming-vs-machine-learning)
3.  [AI vs ML vs Deep Learning](#3-ai-vs-ml-vs-deep-learning)
4.  [Types of Machine Learning](#4-types-of-machine-learning)
5.  [Steps in Building a Machine Learning
    Model](#5-steps-in-building-a-machine-learning-model)
6.  [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7.  [Data Cleaning](#7-data-cleaning)
8.  [Data Preprocessing](#8-data-preprocessing)
9.  [Feature Scaling](#9-feature-scaling)
10. [Feature Engineering](#10-feature-engineering)
11. [Feature Selection](#11-feature-selection)

------------------------------------------------------------------------

## 1. What is Machine Learning?

**Definition:** Machine Learning (ML) is a branch of Artificial
Intelligence (AI) that enables computers to learn patterns from data and
improve their performance on a task without being explicitly programmed
with every rule.

Instead of writing instructions for every possible situation, we provide
a machine-learning algorithm with data. The algorithm identifies
patterns and builds a **model** that can make predictions or decisions
on new data.

### Real-life examples

1.  **YouTube recommendations:** Learns from your watch history, likes,
    and viewing habits to recommend videos.
2.  **Spam detection in Gmail:** Learns patterns in emails to identify
    likely spam.
3.  **Voice assistants:** Systems such as Siri and Alexa use
    machine-learning techniques to process speech and respond.
4.  **Self-driving cars:** Use models to identify road signs,
    pedestrians, lanes, and other vehicles.
5.  **Face unlock:** Learns or matches facial features to recognize an
    authorized user.

## 2. Traditional Programming vs Machine Learning

  -----------------------------------------------------------------------
  Traditional programming             Machine learning
  ----------------------------------- -----------------------------------
  A programmer writes explicit rules. The algorithm learns patterns from
                                      examples.

  Input: data + rules/program.        Training input: data + (often)
                                      known answers; the algorithm learns
                                      a model.

  Output: result.                     Output: a trained model that can
                                      predict results for new data.

  Best when rules are clear and       Useful when patterns are complex or
  stable.                             difficult to describe manually.
  -----------------------------------------------------------------------

**Simple flow**

-   Traditional programming: `Data + Rules → Output`
-   Machine learning:
    `Data + Expected Results → Learning Algorithm → Model`

The model is then used to make predictions on new, unseen data.

## 3. AI vs ML vs Deep Learning

### Artificial Intelligence (AI)

**Definition:** Artificial Intelligence is the broad field of creating
computer systems that can perform tasks that normally require human
intelligence, such as reasoning, learning, understanding language,
recognizing images, and making decisions.

**Examples:** - A computer playing chess. - A system answering
questions. - A robot navigating an environment.

### Machine Learning (ML)

**Definition:** Machine Learning is a subset of AI in which computers
learn patterns from data and use those patterns to make predictions or
decisions.

**Examples:** - YouTube video recommendations. - Netflix
recommendations. - Email spam filtering.

### Deep Learning (DL)

**Definition:** Deep Learning is a subset of Machine Learning that uses
multi-layered artificial neural networks to learn complex patterns from
large amounts of data. These networks are inspired loosely by the
structure of the human brain.

**Examples:** - Face recognition on phones. - Speech recognition. -
Image understanding. - Some conversational AI systems.

### Relationship

`Artificial Intelligence ⊃ Machine Learning ⊃ Deep Learning`

-   **AI** is the broad field of making machines perform intelligent
    tasks.
-   **ML** is one way to achieve AI by learning from data.
-   **DL** is one approach within ML that uses deep neural networks.

## 4. Types of Machine Learning

Machine Learning has three major types:

1.  Supervised Learning
2.  Unsupervised Learning
3.  Reinforcement Learning

### 4.1 Supervised Learning

**Definition:** Supervised Learning is a type of machine learning in
which a model is trained using **labelled data**. Each training example
contains input features and the correct output (label). The model learns
the relationship between them and predicts outputs for new inputs.

**Example:** A model is trained with emails labelled `spam` or
`not spam`. It learns from those examples and classifies a new email.

**Main tasks:**

-   **Classification:** Predicts a category or class. Examples: spam/not
    spam, disease/no disease.
-   **Regression:** Predicts a numerical value. Examples: house price,
    temperature, sales amount.

**Common algorithms:** - Linear Regression - Logistic Regression -
Decision Tree - Naive Bayes - K-Nearest Neighbors (KNN) - Support Vector
Machine (SVM) - Gradient Boosting

> Note: Linear Regression is commonly used for regression; Logistic
> Regression is commonly used for classification.

### 4.2 Unsupervised Learning

**Definition:** Unsupervised Learning is a type of machine learning in
which a model is trained using **unlabelled data**. The data does not
contain correct output labels. The algorithm tries to discover hidden
patterns, structures, or groups in the data.

**Example:** A shop groups customers according to their purchasing
behaviour without being given customer-group labels in advance.

**Common tasks:** - **Clustering:** Groups similar data points together.
Example: customer segmentation. - **Dimensionality reduction:** Reduces
the number of features while preserving important information. Example:
simplifying data for visualization. - **Association:** Finds items or
events that frequently occur together. Example: market-basket analysis.

**Common algorithms:** - K-Means Clustering - Hierarchical Clustering -
DBSCAN - Principal Component Analysis (PCA) - Apriori

### 4.3 Reinforcement Learning

**Definition:** Reinforcement Learning is a type of machine learning in
which an **agent** learns by interacting with an **environment**. It
takes actions, receives rewards or penalties, and learns a strategy
(policy) that aims to maximize the total reward over time.

**Main terms:** - **Agent:** The learner or decision-maker. -
**Environment:** The world in which the agent operates. - **State:** The
current situation. - **Action:** A decision made by the agent. -
**Reward:** Feedback received after an action. - **Policy:** The
strategy used to choose actions.

**Example:** A game-playing program tries different moves. Good moves
earn rewards and poor moves may receive penalties. Over many attempts,
it learns a better strategy.

**Applications:** Game playing, robot navigation, and some
resource-allocation or control problems.

## 5. Steps in Building a Machine Learning Model

The following workflow is listed in the notebook:

1.  **Problem Definition:** Clearly state what needs to be predicted or
    solved. Identify the target, users, constraints, and success
    measure.
2.  **Data Collection:** Gather relevant data from files, databases,
    sensors, APIs, surveys, or other sources.
3.  **Exploratory Data Analysis (EDA):** Inspect and summarize the data
    to understand its structure, patterns, quality, and relationships.
4.  **Data Preprocessing / Cleaning:** Fix or handle missing values,
    duplicates, inconsistent data, incorrect types, and other
    data-quality issues.
5.  **Feature Selection and Engineering:** Select useful input columns
    and create or transform features that help the model learn.
6.  **Split the Dataset:** Divide data into training and testing sets; a
    validation set may also be used. Keep test data separate until final
    evaluation.
7.  **Model Selection:** Choose an algorithm suitable for the problem
    and data.
8.  **Model Training:** Fit the selected algorithm using the training
    data.
9.  **Model Evaluation:** Measure performance using suitable metrics,
    such as accuracy, precision, recall, F1-score, MAE, or RMSE.
10. **Hyperparameter Tuning:** Search for better settings of the model
    that are not learned directly from the training data.
11. **Model Testing / Validation:** Check performance on data not used
    to fit the model and confirm that it generalizes to new examples.

The workflow is iterative: evaluation may show that the data, features,
or model needs improvement.

## 6. Exploratory Data Analysis (EDA)

**Definition:** Exploratory Data Analysis (EDA) is the process of
examining, summarizing, and visualizing a dataset before building a
model. It helps us understand the data, discover patterns, identify
anomalies, detect data-quality problems, and decide the next steps.

Think of EDA as detective work: investigate the data before building a
model or applying transformations.

### Common first-look commands (Python / pandas)

``` python
import pandas as pd

df = pd.read_csv("data.csv")

df.head()       # First five rows
df.tail()       # Last five rows
df.shape        # Number of rows and columns
df.info()       # Column names, data types, non-null counts
df.describe()   # Summary statistics for numeric columns
```

### EDA steps

#### 1. Viewing the data

-   `head()` and `tail()` show sample rows.
-   `shape` gives the number of rows and columns.
-   `info()` shows column names, data types, and non-null counts.
-   Check what columns are available and what type of data each column
    contains.

#### 2. Summary statistics

Common statistics include: - **Mean:** Arithmetic average. - **Median:**
Middle value after sorting. - **Mode:** Most frequent value. -
**Standard deviation (std):** Measures how spread out values are around
the mean. - **Minimum and maximum:** Smallest and largest values. -
**Quartiles:** Values dividing ordered data into four parts.

These statistics help describe the centre and spread of numerical data.

``` python
df.describe()
df["column_name"].mean()
df["column_name"].median()
df["column_name"].mode()
```

#### 3. Value counts

Count how often each unique value appears. This is especially useful for
categorical columns.

``` python
df["category"].value_counts()
```

#### 4. Missing-values analysis

Find how many values are missing in each column and what percentage of
the column is missing.

``` python
df.isnull().sum()
(df.isnull().mean() * 100).round(2)
```

#### 5. Visualizations

  -----------------------------------------------------------------------
  Visualization                       What it helps show
  ----------------------------------- -----------------------------------
  Histogram                           Distribution of a numerical
                                      variable

  Box plot                            Spread, median, and possible
                                      outliers

  Bar plot                            Comparison of categories

  Correlation heatmap                 Strength and direction of
                                      relationships between numerical
                                      features

  Scatter plot                        Relationship between two numerical
                                      variables
  -----------------------------------------------------------------------

#### 6. Target-variable exploration

The **target variable** is the output the model is intended to predict.
Examine its distribution and relationship with other features. For
example, if the target is house price, investigate how price relates to
area, location, and number of rooms.

### Importance of EDA

-   You cannot clean or preprocess data properly without first
    understanding it.
-   It helps identify mistakes, bias, missing information, and
    limitations.
-   It guides data cleaning and feature engineering.
-   It helps explain what the dataset contains and what it may be able
    to tell us.

## 7. Data Cleaning

**Definition:** Data Cleaning is the process of detecting and correcting
or removing inaccurate, incomplete, duplicate, inconsistent, or
irrelevant data so that the dataset is more reliable.

### 1. Handle missing values

First check which columns contain missing values:

``` python
df.isnull().sum()
```

Common strategies:

-   **Drop rows:** Remove rows with missing values when only a small
    number are affected and removing them is reasonable.
-   **Drop columns:** Remove a column when too much information is
    missing and the column is not useful.
-   **Mean imputation:** Fill missing numerical values with the mean;
    can be affected by outliers.
-   **Median imputation:** Fill missing numerical values with the
    median; often more robust when data is skewed or contains outliers.
-   **Mode imputation:** Fill missing categorical values with the most
    frequent value.
-   **Advanced imputation:** Use methods such as KNN or regression-based
    imputation when appropriate.

Example:

``` python
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna(df["city"].mode()[0])
```

### 2. Remove duplicates

Duplicate rows can distort analysis and model training. Detect and
remove exact duplicate rows when they are not valid repeated
observations.

``` python
df.duplicated().sum()
df = df.drop_duplicates()
```

### 3. Fix data types

Convert columns to suitable types. For example, numbers accidentally
stored as strings should be converted to numeric values, and date text
should be parsed as dates.

``` python
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
```

### 4. Handle inconsistent categories

Standardize category values that mean the same thing but are written
differently.

Example: - `male`, `Male`, `MALE` → `Male` - `yes`, `Yes`, `Y` → `Yes`

``` python
df["gender"] = df["gender"].str.strip().str.lower()
df["gender"] = df["gender"].replace({"m": "male", "f": "female"})
```

Check the actual values before applying replacements so that valid
categories are not changed incorrectly.

### 5. Detect and handle outliers

**Definition:** An outlier is a value that is unusually far from most
other values in a dataset.

Methods to investigate outliers: - Box plots - IQR (Interquartile
Range) - Z-score

Possible treatments: - Correct values that are clearly data-entry
errors. - Remove invalid observations when justified. - Cap extreme
values (for example, at selected percentiles) when appropriate. - Keep
genuine extreme observations if they represent real events.

An outlier should not be removed automatically; first investigate
whether it is an error or a meaningful observation.

### 6. Fix domain-specific errors

Use knowledge of the subject area to identify impossible or suspicious
values. For example, a negative age is invalid. Decide whether to
correct, mark as missing, or remove such values based on the source and
context.

**Remember:** EDA helps you discover what is wrong; data cleaning is the
process of addressing those problems. Cleaning is an important part of
real-world machine-learning work.

## 8. Data Preprocessing

**Definition:** Data Preprocessing is the process of transforming raw
data into a clean, consistent, and suitable format for analysis and
machine-learning algorithms.

Data cleaning focuses on fixing data-quality problems. Preprocessing is
broader: it can also include encoding categories, scaling numerical
features, transforming distributions, and preparing data for a model.

### 1. Encoding categorical variables

Machine-learning algorithms generally need numerical inputs.
**Encoding** converts category labels (such as `male`, `yes`, or
`Southeast`) into numerical representations.

#### Label Encoding / Ordinal Encoding

Assigns an integer to each category. Ordinal encoding is suitable when
categories have a meaningful order, such as `Low`, `Medium`, and `High`.

Example: - Low → 0 - Medium → 1 - High → 2

Do not use arbitrary integer labels for nominal categories if the
numbers could wrongly imply an order.

#### One-Hot Encoding

Creates a separate binary column for each category. It is commonly used
for nominal categories that have no natural order, such as region.

Example:

  Region     North   South   West
  -------- ------- ------- ------
  North          1       0      0
  South          0       1      0
  West           0       0      1

### 2. Feature transformation

**Definition:** Feature transformation changes the representation or
distribution of a feature to make it more suitable for analysis or model
training.

Examples: - Log transformation can reduce strong right skew. -
Square-root transformation can reduce some forms of skew. - Other
transformations may be selected according to the data and model.

Use transformations carefully and fit learned preprocessing steps on
training data only to avoid data leakage.

## 9. Feature Scaling

**Definition:** Feature Scaling is the process of bringing numerical
features to comparable scales. It is especially useful for algorithms
that depend on distances, gradients, or feature magnitudes.

### 1. Normalization (Min-Max Scaling)

Scales values to a selected range, commonly 0 to 1.

Formula:

\[ X\_{scaled} = `\frac{X - X_{min}}{X_{max} - X_{min}}`{=tex} \]

### 2. Standardization (Z-score Scaling)

Transforms values so that the feature has a mean of approximately 0 and
a standard deviation of approximately 1.

Formula:

\[ Z = `\frac{X - \mu}{\sigma}`{=tex} \]

Where: - (X) = original value - (`\mu`{=tex}) = mean of the feature -
(`\sigma`{=tex}) = standard deviation of the feature

**Important:** Calculate scaling parameters from the training set and
apply the same fitted transformation to validation and test data.

## 10. Feature Engineering

**Definition:** Feature Engineering is the process of creating new
features from existing data or transforming existing features so that
useful patterns become easier for a machine-learning model to learn.

Why is it needed?

Machine-learning models do not automatically understand every piece of
domain knowledge. Useful features can provide clearer signals and
improve model performance.

### Common feature-engineering techniques

1.  **Mathematical combinations:** Add, subtract, multiply, or divide
    existing numerical features when meaningful.
2.  **Target-based features:** Create features using information related
    to the target only when it is available at prediction time. Avoid
    target leakage.
3.  **Binning:** Convert a continuous variable into ranges or groups
    when useful. Example: age groups.
4.  **Time-based features:** Extract year, month, day, weekday, or hour
    from a date/time column.

## 11. Feature Selection

**Definition:** Feature Selection is the process of choosing the most
useful input features for a machine-learning model and removing
irrelevant, redundant, or less useful features.

### Why is feature selection important?

-   Reduces noise and the risk of overfitting.
-   Can speed up model training.
-   May improve model performance.
-   Makes the model easier to interpret.

### 1. Filter methods (purely statistical)

These methods score features using statistical measures, generally
before fitting a particular predictive model.

-   **Correlation:** Helps identify strongly related numerical features.
    Highly correlated input features may contain redundant information.
-   **Chi-Square test:** Can assess association between categorical
    features and a categorical target, subject to its assumptions.
-   **ANOVA F-test:** Can assess whether a numerical feature differs
    across groups of a categorical target.

### 2. Embedded methods

These methods perform feature selection as part of model training.

-   **Lasso Regression (L1 regularization):** Can shrink some
    coefficients to exactly zero, effectively removing those features
    from the model.
-   **Tree-based models:** Decision trees, Random Forest, and XGBoost
    can provide feature-importance measures.

Feature importance should be interpreted carefully; it does not
automatically prove that a feature causes the target.



