# Conflict Data Analysis Project

This project focuses on analyzing conflict data in Kenya to predict and understand the factors contributing to conflicts. The dataset contains information about various conflicts, including their types, locations, and fatalities.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Data Visualization](#data-visualization)
7. [Modeling](#modeling)
8. [Model Evaluation](#model-evaluation)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Conclusion](#conclusion)

## Introduction

The goal of this project is to analyze conflict data to identify patterns and predict conflict occurrences. By understanding the factors leading to conflicts, we can better inform policies and interventions aimed at reducing violence.

## Setup

### Libraries

The following libraries are required:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)
%matplotlib inline
```

### Data

The dataset used in this project is `conflict_data_ken.csv`.

```python
# Load data
df = pd.read_csv("conflict_data_ken.csv")
df.head()
```

## Data Preprocessing

### Initial Data Inspection

```python
df.info()
df.shape
df.isnull().sum()
```

### Data Cleaning and Selection

```python
# Select relevant columns
columns_to_keep = ['year', 'type_of_violence', 'conflict_name', 'conflict_new_id', 'dyad_new_id', 'side_a', 'side_b', 'side_a_new_id', 'side_b_new_id', 'adm_1', 'adm_2', 'latitude', 'longitude', 'date_start', 'date_end', 'deaths_a', 'deaths_b', 'deaths_civilians', 'deaths_unknown']
df = df[columns_to_keep]

# Drop unnecessary columns
columns_to_drop = ['adm_1', 'adm_2']
df.drop(columns=columns_to_drop, inplace=True)

# Drop the first row (example)
index_label_to_remove = 0
df.drop(index_label_to_remove, inplace=True)
```

## Feature Engineering

### Creating New Features

```python
# Conflict frequency
df['conflict_frequency'] = df.groupby(['side_a', 'side_b'])['conflict_name'].transform('count')

# Conflict intensity
df['conflict_intensity'] = df['deaths_a'] + df['deaths_b'] + df['deaths_civilians']

# Conflict indicator
conditions = [
    (df['deaths_a'] > 5) | (df['deaths_b'] > 5) | (df['deaths_civilians'] > 1)
]
df['conflict_indicator'] = 0
for condition in conditions:
    df.loc[condition, 'conflict_indicator'] = 1
```

## Exploratory Data Analysis

### Basic Analysis

```python
# Summary statistics
df.describe()

# Check for missing values
df.isnull().sum()

# Visualize data distributions
df.hist(figsize=(20, 16), grid=True)
```

### Visualizing Conflict Indicators

```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df['conflict_indicator'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Conflict Indicator')
ax[0].set_ylabel('')
sns.countplot(data=df, x='conflict_indicator', ax=ax[1])
ax[1].set_title('Conflict Indicator')
plt.show()
```

## Data Visualization

### Conflict Frequency and Intensity

```python
plt.figure(figsize=(10, 6))
sns.countplot(x="conflict_name", data=df, palette="flare")
plt.title("Frequency of Conflicts")
plt.xlabel("Conflict Name")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

crosstab = pd.crosstab(df['deaths_a'], df['conflict_indicator'])
crosstab.plot(kind="bar", figsize=(10, 6))
plt.title('Deaths (a) vs Conflict Indicator')
plt.xlabel('Deaths (a)')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
```

## Modeling

### Data Preparation

```python
# Feature and target separation
X = df[['conflict_new_id', 'dyad_new_id', 'side_a_new_id', 'side_b_new_id', 'deaths_a', 'deaths_b', 'deaths_civilians', 'deaths_unknown', 'conflict_frequency', 'conflict_intensity']]
y = df['conflict_indicator']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Logistic Regression Model

```python
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(classification_report(y_test, y_pred))
```

## Model Evaluation

### Confusion Matrix and ROC Curve

```python
# Confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred > 0.5)
confusion_df = pd.DataFrame(confusion_mat, index=['Non-conflict', 'Conflict'], columns=['Non-conflict', 'Conflict'])
sns.heatmap(confusion_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```

## Hyperparameter Tuning

### Random Forest Tuning

```python
rf_params = {"n_estimators": [100, 200, 500, 1000], "max_features": [3, 5, 7], "min_samples_split": [2, 5, 10, 30], "max_depth": [3, 5, 8, None]}
rf_model = RandomForestClassifier(random_state=12345)
gs_cv = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
rf_tuned = RandomForestClassifier(**gs_cv.best_params_).fit(X, y)
cross_val_score(rf_tuned, X, y, cv=10).mean()

# Feature importance
feature_imp = pd.Series(rf_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Variable Severity Levels")
plt.show()
```

## Conclusion

This project demonstrates the process of analyzing conflict data, from data preprocessing and feature engineering to model building and evaluation. The insights gained from this analysis can help inform strategies to mitigate conflicts in Kenya.
