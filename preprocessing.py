import pandas as pd
import numpy as np

df = pd.read_csv("german_credit_data.csv")

df.rename(columns={"Unnamed: 0": "ID"}, inplace=True)

df["Saving accounts"].fillna("no_info", inplace=True)
df["Checking account"].fillna("no_info", inplace=True)

df = pd.get_dummies(df, drop_first=True)

df = df.astype(int)

print(df.head())
print(df.columns.tolist())
