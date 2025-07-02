import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# ------------------------------
# Function: Load the data
# ------------------------------
def load_data(path, target_column):
    """
    Load data from a CSV file and separate the target column.

    Parameters:
    path (str): Path to the CSV file.
    target_column (str): Name of the target variable column.

    Returns:
    tuple: (df, y) where df is the DataFrame and y is the target variable.
    """
    df = pd.read_csv(path)
    y = df[target_column]
    return df, y

# ------------------------------
# Function: Plot missing values
# ------------------------------
def plot_missing_values(df):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    msno.bar(df, ax=axs[0])
    axs[0].set_title("Distribution of Missing Values")
    msno.matrix(df, ax=axs[1])
    axs[1].set_title("Matrix of Missing Values")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Function: Percentage of missing values
# ------------------------------
def percentage_of_missing_values(df):
    percentage = df.isna().mean() * 100
    print("Percentage of missing values per column:")
    return percentage

# ------------------------------
# Function: Plot target variable
# ------------------------------
def plot_target_variable(y):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title("Distribution of Target Variable")
    plt.xlabel("Target Variable")
    plt.ylabel("Count")
    plt.grid(False)
    plt.show()

# ------------------------------
# Function: Check duplicates
# ------------------------------
def check_duplicates(df):
    count = df.duplicated().sum()
    if count > 0:
        print(f"Data has {count} duplicate rows.")
    else:
        print("No duplicates found.")

# ------------------------------
# Function: Detect outliers
# ------------------------------
def detect_outliers(df, y):
    features = [col for col in df.columns if col != y.name]
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    df[features].plot(kind='box', subplots=True, layout=(n_rows, n_cols),
                      figsize=(6*n_cols, 4*n_rows), sharex=False, sharey=False)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Function: Fill missing values
# ------------------------------
def fill_missing_values(df, missing_dict):
    df_filled = df.copy()
    for column, method in missing_dict.items():
        if method == 'mean':
            df_filled[column] = df_filled[column].fillna(df_filled[column].mean())
        elif method == 'median':
            df_filled[column] = df_filled[column].fillna(df_filled[column].median())
        elif method == 'mode':
            df_filled[column] = df_filled[column].fillna(df_filled[column].mode().iloc[0])
        elif method in ['ffill', 'bfill']:
            df_filled[column] = df_filled[column].fillna(method=method)
        else:
            df_filled[column] = df_filled[column].fillna(method)
    return df_filled

# ------------------------------
# Function: Clean data
# ------------------------------
def cleaning_data(df, rename_dict=None, columns_to_drop=None,
                  drop_duplicates=True, missing_dict=None):
    df_cleaned = df.copy()
    if rename_dict:
        df_cleaned.rename(columns=rename_dict, inplace=True)
    if columns_to_drop:
        df_cleaned.drop(columns=columns_to_drop, inplace=True)
    if drop_duplicates:
        df_cleaned.drop_duplicates(inplace=True)
    if missing_dict:
        df_cleaned = fill_missing_values(df_cleaned, missing_dict)
    return df_cleaned

# ------------------------------
# Function: Analyze data
# ------------------------------
def analyze_data(df, y):
    print("Preview of DataFrame:")
    print(df.head())
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print("-" * 40)
    print("Descriptive Statistics:")
    print(df.describe().T)
    print("-" * 40)
    print("Unique values per column:")
    print(df.nunique())
    print("-" * 40)
    check_duplicates(df)
    print("-" * 40)
    plot_missing_values(df)
    print("-" * 40)
    print(percentage_of_missing_values(df))
    print("-" * 40)
    plot_target_variable(y)
    print("-" * 40)
    print("Target variable counts:")
    print(y.value_counts())
    print("-" * 40)
    detect_outliers(df, y)
# ------------------------------
# Function: plotting data
# ------------------------------

def plot_bank_accounts():
    countries = ['Uganda', 'Kenya', 'Tanzania', 'Rwanda']
    bank_accounts = [181, 1521, 607, 1003]
    total_population = [2101, 6068, 6620, 8735]

    percentages = [round((ba / tp) * 100, 2) for ba, tp in zip(bank_accounts, total_population)]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(countries, bank_accounts, color='skyblue', label='Number of people with bank account')
    bars2 = ax.bar(countries, total_population, color='lightgreen', alpha=0.5, label='Total number of people')

    for bar, bank_count, total_count, percentage in zip(bars1, bank_accounts, total_population, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, f'{bank_count} ({percentage}%)', ha='center', va='bottom')
        ax.text(bar.get_x() + bar.get_width() / 2, total_count + 10, f'{total_count}', ha='center', va='bottom')

    ax.set_xlabel('Countries')
    ax.set_ylabel('Number of People')
    ax.set_title('Number of People with Bank Account and Total Population by Country')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load your data - update filename and target column name as needed
    df, y = load_data("your_data.csv", "target")  
    
    # Optional: clean your data before analysis, adjust parameters as needed
    df = cleaning_data(df, columns_to_drop=["unwanted_column"], missing_dict={"age": "median"})
    
    # Run the full analysis
    analyze_data(df, y)
    
    # Also show your bank accounts plot
    plot_bank_accounts()
