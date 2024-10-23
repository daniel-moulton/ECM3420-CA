import pandas as pd

df = pd.read_csv("imdb-movies-dataset.csv")

# Drop the superfluous columns
df = df[["Title", "Director", "Rating", "Cast", "Votes"]]

# Drop duplicates
df = df.drop_duplicates()

# Print the rows with missing values
# print(df[df.isna().any(axis=1)])


# Drop rows with missing values
df = df.dropna()


# Convert the Votes column to an integer
df["Votes"] = df["Votes"].str.replace(",", "").astype(int)

# Work out the bottom 10% of films by votes
bottom_10_percent = df["Votes"].quantile(0.10)

print(bottom_10_percent)

# Filter out the bottom 10% of films by votes
df = df[df["Votes"] > bottom_10_percent]

# Print the cleaned data



# Save the cleaned data
df.to_csv("imdb-movies-dataset-cleaned.csv", index=False)

