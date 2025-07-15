import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# Set seaborn style
sns.set(style = "whitegrid", context = "talk")

# Set a Larger font and a color palette
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 12
})

#Load the Dataset
df = pd.read_csv("Iris.csv")
print(df.head()) #first 5 Rows
print(df.describe()) # shows(count,mean,std,min,max,25%,50%,75%)
print(df.columns) # Print every column name 
print(df.isna().sum()) #Checking for NaN values
print(df['Species'].value_counts()) #Number of each species 
df.shape
print(df)

# Doing Comparison of Sepal lenght v/s Sepal Width
sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=df)
plt.title('Sepal Length v/s Sepal Width Comparison')
plt.show()

# Bar chart showing count of each species
plt.figure(figsize = (8, 5))
colors = sns.color_palette("coolwarm", 3)
ax = sns.countplot(x = "Species" , data = df, palette = colors, edgecolor = "black" )

# Beautify chart
plt.title("Count Of Each Iris Species", fontsize=16, weight='bold')
plt.ylabel("Count", fontsize=13)
plt.xlabel("Species", fontsize=13)
plt.tight_layout()
plt.show()

# Prepare data
mean_values = df.groupby("Species")[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].mean().reset_index()
melted_df = mean_values.melt(id_vars = "Species", var_name = "Feature", value_name = "Mean")

#Plot
plt.figure(figsize = (10,6))
ax = sns.barplot(x = "Feature", y = "Mean", hue = "Species", data = melted_df, palette = "pastel", edgecolor = "gray")

# Style it up
plt.title("Average Feature Values per Iris Species", fontsize = 16, weight = 'bold')
plt.ylabel("Mean Value (cm)")
plt.xlabel("Feature")
plt.grid(axis = 'y', linestyle = '--', alpha = 0.6)
plt.tight_layout()
plt.show()

# Plot Histograms for all numeric features
df.hist(figsize = (10, 8), bins = 20, color = 'lightseagreen', edgecolor = 'black')
plt.suptitle("Distribution of Features in Iris Dataset (Histograms)", fontsize = 16, weight = 'bold')
plt.tight_layout()
plt.subplots_adjust(top = 0.92) # Adjust to make space for title
plt.show()



