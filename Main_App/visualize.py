import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

# دالة لتحميل البيانات
def load_data(file_path):
    return pd.read_csv(file_path)


# دالة لعمل Histogram
def plot_histogram(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    return fig

# دالة لعمل Density Plot
def plot_density(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df[column], ax=ax)
    ax.set_title(f"Density Plot of {column}")
    return fig

# دالة لعمل Boxplot
def plot_boxplot(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")
    return fig

# دالة لعمل Violin Plot
def plot_violin(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x=df[column], ax=ax)
    ax.set_title(f"Violin Plot of {column}")
    return fig

# دالة لعمل Bar Plot
def plot_bar(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=df[column].value_counts().index, y=df[column].value_counts().values, ax=ax)
    ax.set_title(f"Bar Plot of {column}")
    return fig

# دالة لعمل Pie Chart
def plot_pie(df, column):
    fig, ax = plt.subplots(figsize=(8, 8))
    df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, cmap="Set3")
    ax.set_title(f"Pie Chart of {column}")
    return fig

# دالة لعمل Scatter Plot
def plot_scatter(df, x_column, y_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_column], df[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
    return fig

# دالة لعمل Correlation Heatmap
def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

# دالة لعمل Pairplot
def plot_pairplot(df):
    fig = sns.pairplot(df)
    fig.fig.suptitle("Pairplot", y=1.02)
    return fig

# دالة لعمل Line Plot (Time Series)
def plot_line(df, x_column, y_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=df[x_column], y=df[y_column], ax=ax)
    ax.set_title(f"Line Plot: {x_column} vs {y_column}")
    return fig

# دالة لعمل Missing Data Heatmap
def plot_missing_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Data Heatmap")
    return fig

# دالة لعمل Missing Data Barplot
def plot_missing_barplot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.isnull().sum().plot.bar(ax=ax)
    ax.set_title("Missing Data Barplot")
    return fig

# دالة لعمل Word Cloud
def plot_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")
    return fig
