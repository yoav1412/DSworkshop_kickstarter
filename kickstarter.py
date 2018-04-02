import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime as dt
import os

def date_parser(date_str):
    try:
        res = dt.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return res
    except ValueError:
        pass
    try:
        res = dt.strptime(date_str, '%Y-%m-%d')
        return res
    except ValueError:
        raise


def map_percetage_to_classes(perc):
    """
    :param perc: percentage of pledged money out goal amount (usd)
    :return: encoding of a prediction class (bucketization of numeric variable)
    """
    if perc >= 3:
        return 4
    if 1 <= perc < 3 :
        return 3
    if 0.8 <= perc < 1:
        return 2
    if perc < 0.8:
        return 1


df = pd.DataFrame.from_csv(os.getcwd() + "\\kickstarterData\\ks-projects-201801.csv")
# Fix broken name of column:
df = df.rename(columns={"usd pledged": "usd_pledged"})

# Dummify categorical explanatory variables:
df = pd.get_dummies(df, columns = ['category', 'main_category', 'currency','country'])

# Encode "state" variable: TODO: this encoding is bad (ignores some buckets), fix or remove if we don't use "state"
df.state = df.state.apply(func=lambda x: 1 if x=="successful" else 0)

# Create new column, % of goal taht has been pledged:
df["percentage_pledged"] = df.apply(lambda row: row.usd_pledged / row.usd_goal_real, axis=1)

# Create new column, length (in days) of time from launch of project to deadline:
df["deadline_days"] = df.apply(lambda row:(date_parser(row.deadline) -
                                          date_parser(row.launched)).days, axis = 1)
# Trreat 0 deadline_days as 1 (rare case, 102 times):
df["deadline_days"] = df.apply(lambda row: 1 if row.deadline_days == 0 else row.deadline_days, axis = 1)
# Remove deadline_days outliers (4 cases of deadline_days > 10000):
df = df.loc[df['deadline_days'] < 10000]

# Drop unused columns:
df = df.drop(axis=1, labels=["name", "deadline", "launched"])
# Drop columns that are irrelevant as predictive variables (they are not available in start of KickStarter project):
df = df.drop(axis=1, labels=['pledged', 'usd_pledged', 'usd_pledged_real', 'backers', 'state'])
# Drop nan values from specific column: TODO: write which column and how many rows were removed
df = df.dropna()

# Remove outliers from percentage_pledged column:
df2 = df.loc[df['percentage_pledged'] < np.percentile(df.percentage_pledged.values, 95)]
df2 = df2.loc[df2['percentage_pledged'] > np.percentile(df.percentage_pledged.values, 5)]
df = df2

# Create pledged_classes column, a bucketization of "percentage_pledged" column:
df["pledged_classes"] = df.apply(func = lambda x: map_percetage_to_classes(x.percentage_pledged), axis=1)
# percentage_pledged column no longer needed:
df = df.drop(axis=1, labels=["percentage_pledged"])

# Partition data to train/test sets:
PREDICTED_VRIABLE_COL_NAME = "pledged_classes"
train_X, test_X, train_y, test_y = train_test_split(df.drop(labels=[PREDICTED_VRIABLE_COL_NAME], axis=1),
                                                    df[PREDICTED_VRIABLE_COL_NAME],
                                                    random_state=7777)

# Fit Model:
logRegClf = LogisticRegression()
linReg = LinearRegression()
RandForestRegressor = RandomForestRegressor(n_estimators=100, max_depth=None)
RandForestClf = RandomForestClassifier(n_estimators=20, max_depth=6)

RandForestClf.fit(train_X, train_y)
print(RandForestClf.score(test_X, test_y))