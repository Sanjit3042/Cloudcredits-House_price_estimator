# house_price_estimator.py

# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 2. Load the dataset (replace 'house_data.csv' with your actual file)
data = pd.read_csv(r'C:\Users\user\Downloads\CLOUDCREDITS\House_price_estimator\data\house_data.csv')

# 3. Basic data exploration
print(data.info())
print(data.describe())

# 4. Handle missing values and encode categorical features
# Let's assume columns: 'location' should be 'area' and other columns are as per the data
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# 5. Preprocessing pipeline for numerical and categorical data remains the same
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# 6. Split the data into features and target
X = data.drop('price', axis=1)
y = data['price']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Model pipeline (can choose between models)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Can swap with LinearRegression() if needed
])

# 9. Train the model
model_pipeline.fit(X_train, y_train)

# 10. Predict and evaluate
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# 11. Visualize Actual vs Predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# 12. Feature Importance (only applicable for tree-based models like RandomForest)
if isinstance(model_pipeline.named_steps['regressor'], RandomForestRegressor):
    importance = model_pipeline.named_steps['regressor'].feature_importances_
    feature_names = numerical_features + list(model_pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out())
    
    # Plot feature importance
    sns.barplot(x=importance, y=feature_names)
    plt.title('Feature Importance')
    plt.show()

# 13. Distribution of house prices
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
