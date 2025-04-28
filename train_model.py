import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np

# Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("crime_data.csv")

# Selecting features
features = ['Crime_Type', 'Latitude', 'Longitude', 'Date', 'Time', 'Weather', 'Population_Density']
X = df[features]
y = df['Crime_Type']  # Target variable

# Convert categorical features to category type
cat_cols = ['Crime_Type', 'Population_Density', 'Weather']
for col in cat_cols:
    X[col] = X[col].astype('category')

# Convert 'Date' to numerical features
X['Date'] = pd.to_datetime(X['Date'])
X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day
X['DayOfWeek'] = X['Date'].dt.weekday

# Convert 'Time' to hour-of-the-day (0-23)
X['Time'] = pd.to_datetime(X['Time'], format='%H:%M').dt.hour

# Drop original 'Date' column
X.drop(columns=['Date'], inplace=True)

# Convert all columns to numeric before splitting
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# One-hot encode 'Weather'
X = pd.get_dummies(X, columns=['Weather'], drop_first=True)

# Encode categorical labels properly
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert 'Population_Density' properly
X['Population_Density'] = X['Population_Density'].astype(str)  # Convert to string
X['Population_Density'] = X['Population_Density'].astype('category').cat.codes

# Fill missing values properly
cat_cols = X.select_dtypes(include=['category']).columns
num_cols = X.select_dtypes(include=['number']).columns

X[num_cols] = X[num_cols].fillna(0)  # Fill numerical columns with 0

for col in cat_cols:
    print(f"Column: {col}, Unique Values: {X[col].unique()}")
    X[col] = X[col].cat.add_categories(['Unknown']).fillna('Unknown')  # Handle missing categorical values
    X[col] = X[col].cat.codes  # Convert categorical data to numerical

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure consistency in train and test sets
print(f"X_train columns: {X_train.columns}")
print(f"X_test columns: {X_test.columns}")

# Standardizing numerical columns
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=['number']).columns  # Ensure only numerical data
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
try:
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # Save model
    joblib.dump(model, "models/crime_model.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model and encoders saved successfully!")

except Exception as e:
    print(f"Error during training: {e}")

# Model evaluation
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")