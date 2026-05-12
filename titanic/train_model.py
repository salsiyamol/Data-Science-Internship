import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
df = pd.read_csv('titanic.csv')

# 2. Basic Preprocessing
# Fill missing age values with the median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Convert Sex to numbers: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Choose features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

# 3. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save the model using joblib
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as 'model.pkl'!")
