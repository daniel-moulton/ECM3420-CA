import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the cleaned data
df = pd.read_csv("imdb-movies-dataset-cleaned.csv")

# Group the data by Director and calculate average ratings and film count
director_ratings = df.groupby('Director').agg({'Rating': 'mean', 'Director': 'count'}).rename(columns={'Director': 'Film_Count'})

# Set a minimum number of films
min_films = 5
director_ratings = director_ratings[director_ratings['Film_Count'] >= min_films].sort_values('Rating', ascending=False)

# Filter the original dataframe for these directors
df = df[df['Director'].isin(director_ratings.index)]
df['Director_encoded'] = df['Director'].astype('category').cat.codes

# Define input and output
X = df[['Director_encoded']]
y = df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Create a new DataFrame for the results
results = pd.DataFrame({
    'Title': df.loc[X_test.index, 'Title'],
    'Director': df.loc[X_test.index, 'Director'],
    'Actual Rating': y_test,
    'Predicted Rating': y_pred
})

# Print the results
print(results)

# Save the results
results.to_csv("director_results.csv", index=False)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Optionally, plot actual vs predicted ratings
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs. Predicted Ratings by Director')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # 45-degree line
plt.show()
