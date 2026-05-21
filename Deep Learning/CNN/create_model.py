import os
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a small sample dataset to satisfy the model configuration
X_dummy = np.random.rand(10, 784)
y_dummy = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create and fit a lightweight placeholder model
model = LogisticRegression()
model.fit(X_dummy, y_dummy)

# Target the exact path right next to your app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(BASE_DIR, "fashion_model_package.joblib")

# Write the file out securely
with open(save_path, "wb") as f:
    joblib.dump(model, f)

print(f"✅ SUCCESS: Placeholder model file created at: {save_path}")
print(f"File Size: {os.path.getsize(save_path) / 1024:.2f} KB")