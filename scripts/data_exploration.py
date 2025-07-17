# scripts/data_exploration.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data\creditcard.csv')

# Basic info
print(df.shape)
print(df.info())
print(df.isnull().sum())

# Class distribution
print(df['Class'].value_counts())

# Class imbalance plot
sns.countplot(x='Class', data=df)
plt.title('Fraud (1) vs Non-Fraud (0)')
plt.show()
