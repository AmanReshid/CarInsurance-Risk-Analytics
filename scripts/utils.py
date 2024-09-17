import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns['% of Total Values'] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns



def detect_outliers(df, numerical_columns):
    outliers_summary = {}
    
    for column in numerical_columns:
        if column in df.columns:
            # Z-score method
            z_scores = np.abs(zscore(df[column].dropna()))  
            outliers_z = np.where(z_scores > 3)[0]  # Indices of outliers
            
            # Summary of outliers (count and percentage)
            outliers_count = len(outliers_z)
            outliers_percentage = (outliers_count / len(df[column].dropna())) * 100
            outliers_summary[column] = {
                'outliers_count': outliers_count,
                'outliers_percentage': outliers_percentage
            }
            
            # Box plot method
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[column])
            plt.title(f'Box Plot for {column}')
            plt.show()
    
    # Formatting the output for better readability
    formatted_output = "\nOutliers Summary:\n"
    for column, stats in outliers_summary.items():
        formatted_output += (
            f"\nColumn: {column}\n"
            f" - Outliers Count: {stats['outliers_count']}\n"
            f" - Outliers Percentage: {stats['outliers_percentage']:.2f}%\n"
        )
    
    return formatted_output


def convert_bytes_to_megabytes(df, bytes_data):
    megabyte = 1e+6  # 1 MB = 1e+6 Bytes
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]


def convert_ms_to_seconds(ms):
    return ms / 1000



def fix_outlier(df, column, percentile=0.95):
    threshold = df[column].quantile(percentile)
    median = df[column].median()
    df[column] = np.where(df[column] > threshold, median, df[column])
    return df[column]


def remove_outliers(df, column_to_process, z_threshold=3):
    # Apply outlier removal to the specified column
    df = df.copy()  # Avoid modifying the original DataFrame
    z_scores = zscore(df[column_to_process].dropna())
    df['z_score'] = np.nan
    df.loc[df[column_to_process].notna(), 'z_score'] = z_scores

    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(df['z_score']) > z_threshold).astype(int)
    df = df[df[outlier_column] == 0]  # Keep rows without outliers

    # Drop the outlier column as it's no longer needed
    df = df.drop(columns=[outlier_column, 'z_score'], errors='ignore')

    return df

def plot_histograms(df, columns, log_scale_threshold=1000):
    num_cols = len(columns)
    plt.figure(figsize=(15, num_cols * 4))

    for i, col in enumerate(columns):
        plt.subplot(num_cols, 1, i + 1)
        
        # Check if the column contains large values
        if df[col].max() > log_scale_threshold:
            sns.histplot(np.log1p(df[col]), kde=True, bins=30)
            plt.title(f'Log Distribution of {col}', fontsize=16)
            plt.xlabel(f'Log of {col}', fontsize=12)
        else:
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}', fontsize=16)
            plt.xlabel(col, fontsize=12)
        
        plt.ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_bar_charts(df, columns):
    num_cols = len(columns)
    plt.figure(figsize=(15, num_cols * 4))

    for i, col in enumerate(columns):
        plt.subplot(num_cols, 1, i + 1)
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Count of Categories in {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    
    plt.tight_layout()
    plt.show()




def analyze_monthly_changes(df, postal_code_col, month_col, premium_col, claims_col):
    # Step 1: Sort data by PostalCode and TransactionMonth
    df = df.sort_values(by=[postal_code_col, month_col])
    
    # Step 2: Calculate monthly changes in TotalPremium and TotalClaims
    df['MonthlyChange_TotalPremium'] = df.groupby(postal_code_col)[premium_col].diff()
    df['MonthlyChange_TotalClaims'] = df.groupby(postal_code_col)[claims_col].diff()

    # Step 3: Drop rows where changes are NaN (first month in each group)
    df = df.dropna(subset=['MonthlyChange_TotalPremium', 'MonthlyChange_TotalClaims'])

    # Step 4: Scatter plot of monthly changes with PostalCode color-coded
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='MonthlyChange_TotalPremium', y='MonthlyChange_TotalClaims', hue=postal_code_col, palette='viridis', alpha=0.7)
    plt.title('Monthly Changes in Total Premium vs Total Claims by PostalCode')
    plt.xlabel('Monthly Change in Total Premium')
    plt.ylabel('Monthly Change in Total Claims')
    plt.show()

    # Step 5: Calculate correlation between monthly changes for each PostalCode
    correlations = []
    postal_codes = df[postal_code_col].unique()
    
    for code in postal_codes:
        temp_df = df[df[postal_code_col] == code]
        corr = temp_df[['MonthlyChange_TotalPremium', 'MonthlyChange_TotalClaims']].corr().iloc[0, 1]
        correlations.append((code, corr))
    
    # Create a DataFrame with PostalCode and the correlation values
    correlation_df = pd.DataFrame(correlations, columns=[postal_code_col, 'Correlation'])
    
    return correlation_df




def bivariate_analysis(df, x_col, y_col, show_regression_line=False):
    # Check if columns exist in DataFrame
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found in DataFrame.")
    
    # Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5, label='Data points')
    
    # Regression Line (optional)
    if show_regression_line:
        sns.regplot(x=x_col, y=y_col, data=df, scatter=False, color='red')
    
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()
    
    # Correlation Coefficient
    correlation = df[x_col].corr(df[y_col])
    print(f"Correlation between {x_col} and {y_col}: {correlation:.2f}")