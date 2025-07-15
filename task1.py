import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Add flag to separate train/test later
train_df["TrainFlag"] = 1
test_df["TrainFlag"] = 0
test_df["Survived"] = np.nan

# Combine for consistent preprocessing
combined = pd.concat([train_df, test_df], sort=False)

# Fill missing values
combined["Age"] = combined["Age"].fillna(combined["Age"].median())
combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())
combined["Embarked"] = combined["Embarked"].fillna(combined["Embarked"].mode()[0])

# Encode categorical features
# Sex (Gender)
le_sex = LabelEncoder()
combined["Sex"] = le_sex.fit_transform(combined["Sex"])

# Embarked
le_embarked = LabelEncoder()
combined["Embarked"] = le_embarked.fit_transform(combined["Embarked"])

# Drop unused columns (Name, Ticket, Cabin are identifiers/less useful for this simple model)
combined.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Split back into cleaned train and test sets
train_clean = combined[combined["TrainFlag"] == 1].drop("TrainFlag", axis=1)
test_clean = combined[combined["TrainFlag"] == 0].drop(["TrainFlag", "Survived"], axis=1)

# Define input (X) and target (y)
X = train_clean.drop(["Survived", "PassengerId"], axis=1)
y = train_clean["Survived"]

# Feature Scaling for numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch'] # Identify numerical columns for scaling

for col in numerical_features:
    if col in X.columns:
        X[col] = scaler.fit_transform(X[[col]]) # Fit and transform on training data
    if col in test_clean.columns:
        test_clean[col] = scaler.transform(test_clean[[col]]) # Only transform on test data

# Train/test split for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, y_pred))

# Predict on test set
X_test = test_clean.drop(["PassengerId"], axis=1)
test_predictions = model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    "PassengerId": test_clean["PassengerId"],
    "Survived": test_predictions.astype(int)
})

# Drop unused columns (Name, Ticket, Cabin are identifiers/less useful for this simple model)
combined.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True, errors = 'ignore')

# --- View Feature Engineering Results ---
print("--- Viewing Feature Engineering Results ---")

print("\nCombined DataFrame head after all preprocessing:")
print(combined.head())

print("\nCombined DataFrame info after all preprocessing:")
combined.info()

print("\nValue counts for Sex (0/1 after encoding):")
print(combined['Sex'].value_counts())

print("\nValue counts for Embarked (0/1/2 after encoding):")
print(combined['Embarked'].value_counts())

print("\nMissing values remaining in Combined DataFrame:")
print(combined.isnull().sum()) # Should ideally all be 0 for features

print("-----------------------------------------")


# Split back into cleaned train and test sets
train_clean = combined[combined["TrainFlag"] == 1].drop("TrainFlag", axis=1)
test_clean = combined[combined["TrainFlag"] == 0].drop(["TrainFlag", "Survived"], axis=1)

#Survived (Count Plot)
plt.figure(figsize = (6, 4))
sns.countplot(x = 'Survived', data = train_df )
plt.title('Survival Count (0 = Not Survived, 1 = Survived )')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Save to CSV
submission.to_csv("titanic_submission.csv", index=False)
print("Submission file saved as titanic_submission.csv")