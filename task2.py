import pandas as pd

# Load your dataset
df = pd.read_csv("test.csv")  # Make sure test.csv is in the same directory

# Select only numerical columns
numerical_columns = df.select_dtypes(include=["int64", "float64"])

# Calculate summary statistics
summary_stats = {
    "Mean": numerical_columns.mean(),
    "Median": numerical_columns.median(),
    "Mode": numerical_columns.mode().iloc[0],
    "Standard Deviation": numerical_columns.std()
}

# Create a DataFrame to display results neatly
summary_df = pd.DataFrame(summary_stats)

# Show the result
print(summary_df)
