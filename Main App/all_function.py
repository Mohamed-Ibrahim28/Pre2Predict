import pandas as pd
import numpy as np
import datetime
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import io
from sklearn.feature_selection import VarianceThreshold



# Function to load DataFrame in Streamlit app
def load_data(uploaded_file, file_type='csv', **kwargs):
    if uploaded_file is None:
        return None
    try:
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file, **kwargs)
        elif file_type == 'excel':
            df = pd.read_excel(uploaded_file, **kwargs)
        elif file_type == 'json':
            df = pd.read_json(uploaded_file, **kwargs)
        elif file_type == 'parquet':
            df = pd.read_parquet(uploaded_file, **kwargs)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def check_data_integrity(df):
    result = {
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }
    return result



def log_data_metadata(df, uploaded_file=None):
    metadata = {
        'load_time': datetime.datetime.now().isoformat(),
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'column_types': df.dtypes.apply(lambda x: x.name).to_dict(),
        'file_size_kb': len(uploaded_file.getvalue()) / 1024 if uploaded_file else None
    }
    return metadata


def list_all_columns(df):
    columns = df.columns.tolist()
    st.write(f"### Total Columns: {len(columns)}")
    for idx, col in enumerate(columns, 1):
        st.write(f"{idx}. {col}")


def view_column_info(df, column_name):
    if column_name not in df.columns:
        st.warning(f"Column '{column_name}' not found.")
        return

    col_data = df[column_name]
    info = {
        "Data Type": col_data.dtype,
        "Missing Values (%)": col_data.isna().mean() * 100,
        "Unique Values": col_data.nunique(),
        "Sample Values": col_data.dropna().sample(min(5, len(col_data.dropna())), random_state=1).tolist()
    }

    st.write(f"### Information for column '{column_name}':")
    for key, value in info.items():
        st.write(f"- **{key}**: {value}")


def view_all_columns_summary(df):
    summary = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values (%)": df.isna().mean() * 100,
        "Unique Values": df.nunique()
    })
    st.dataframe(summary)
    return summary


def remove_columns(df, columns_to_remove):
    missing_cols = [col for col in columns_to_remove if col not in df.columns]
    if missing_cols:
        st.warning(f"Columns not found and cannot be removed: {missing_cols}")

    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
    return df


def rename_columns(df, rename_dict):
    invalid_keys = [key for key in rename_dict.keys() if key not in df.columns]
    if invalid_keys:
        st.warning(f"Some columns not found and were not renamed: {invalid_keys}")

    df = df.rename(columns=rename_dict)
    return df


def get_missing_columns(df):
    return df.columns[df.isnull().any()].tolist()

def handle_missing_column(df, col, method, custom_value=None):
    if method == "Drop":
        df = df.dropna(subset=[col])
    elif method == "Mean":
        df[col] = df[col].fillna(df[col].mean())
    elif method == "Median":
        df[col] = df[col].fillna(df[col].median())
    elif method == "Mode":
        df[col] = df[col].fillna(df[col].mode()[0])
    elif method == "Custom":
        df[col] = df[col].fillna(custom_value)
    return df

def fill_missing_in_columns(df, columns, fill_value):
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Columns not found: {missing_columns}")

    valid_columns = [col for col in columns if col in df.columns]
    df[valid_columns] = df[valid_columns].fillna(fill_value)
    st.success(f"Filled missing values in {len(valid_columns)} columns with '{fill_value}'.")
    return df


def replace_values_in_columns(df, columns, to_replace, replacement):
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Columns not found: {missing_columns}")

    valid_columns = [col for col in columns if col in df.columns]
    df[valid_columns] = df[valid_columns].replace(to_replace, replacement)
    st.success(f"Replaced '{to_replace}' with '{replacement}' in {len(valid_columns)} columns.")
    return df



def check_duplicates(df):
    """Check if there are duplicate rows."""
    duplicates = df.duplicated()
    num_duplicates = duplicates.sum()
    return num_duplicates

def remove_duplicates(df):
    """Remove duplicate rows and return cleaned df + عدد اللي اتشالو"""
    before = df.shape[0]
    df_cleaned = df.drop_duplicates()
    after = df_cleaned.shape[0]
    removed = before - after
    return df_cleaned, removed

def handle_outliers(df, method='IQR', contamination=0.01):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_outliers_detected = 0
    columns_handled = []

    if method == 'IQR':
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            outliers_in_col = outliers.sum()
            if outliers_in_col > 0:
                total_outliers_detected += outliers_in_col
                columns_handled.append(col)
                df[col] = np.clip(df[col], lower_bound, upper_bound)

    elif method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = (z_scores > 3)
            outliers_in_col = outliers.sum()
            if outliers_in_col > 0:
                total_outliers_detected += outliers_in_col
                columns_handled.append(col)
                df.loc[outliers, col] = df[col].median()

    elif method == 'isolation_forest':
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(df[numeric_cols].fillna(0))
        total_outliers_detected = (preds == -1).sum()
        if total_outliers_detected > 0:
            columns_handled = numeric_cols.tolist()
            df = df[preds == 1]

    else:
        return df

    return df


def encode_features(df, columns, method='label'):
    df = df.copy()

    if method == 'label':
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col].astype(str))
        st.success("Label Encoding applied.")

    elif method == 'onehot':
        df = pd.get_dummies(df, columns=columns)
        st.success("One-Hot Encoding applied.")

    elif method == 'ordinal':
        oe = OrdinalEncoder()
        df[columns] = oe.fit_transform(df[columns].astype(str))
        st.success("Ordinal Encoding applied.")

    else:
        st.error("Invalid method. Use 'label', 'onehot', or 'ordinal'.")

    return df


def save_dataset(df, filename='processed_data.csv', file_format='csv'):
    if file_format == 'csv':
        df.to_csv(filename, index=False)
        st.success(f"Dataset saved as CSV: {filename}")
    elif file_format == 'excel':
        df.to_excel(filename, index=False)
        st.success(f"Dataset saved as Excel: {filename}")
    else:
        st.error("Unsupported format. Use 'csv' or 'excel'.")

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def replace_values_in_columns(df, columns, to_replace, replacement):
    """
    Replaces specific values in selected columns of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        columns (list): List of column names to apply the replacement.
        to_replace (any): The value you want to replace.
        replacement (any): The value to replace with.

    Returns:
        pd.DataFrame: Modified DataFrame.
    """
    # 1. Check for missing columns
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        st.warning(f"These columns were not found and skipped: {missing_columns}")

    # 2. Perform replacement on valid columns
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        st.info("No valid columns selected for replacement.")
        return df

    df[valid_columns] = df[valid_columns].replace(to_replace, replacement)
    return df

def change_columns_dtype(df, columns, new_dtype):
    for column in columns:
        try:
            if new_dtype == "datetime":
                df[column] = pd.to_datetime(df[column], errors='coerce')
            elif new_dtype == "category":
                df[column] = df[column].astype('category')
            else:
                df[column] = df[column].astype(new_dtype)
            st.success(f"✅ '{column}' converted to {new_dtype}")
        except Exception as e:
            st.error(f"❌ Error converting '{column}': {e}")
    return df

def extract_datetime_features(df, column):
    try:
        df[column] = pd.to_datetime(df[column], errors='coerce')
        df[f"{column}_year"] = df[column].dt.year
        df[f"{column}_month"] = df[column].dt.month
        df[f"{column}_day"] = df[column].dt.day
        df[f"{column}_weekday"] = df[column].dt.weekday
        df[f"{column}_hour"] = df[column].dt.hour
        st.success(f"Extracted datetime features from '{column}'.")
    except Exception as e:
        st.error(f"Failed to extract datetime features: {e}")
    return df

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def scale_features(df, columns, method='minmax'):
    df = df.copy()
    scaler = None

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        st.error("Invalid method selected.")

    if scaler:
        df[columns] = scaler.fit_transform(df[columns])
        st.success(f"{method.capitalize()} scaling applied.")

    return df


def generate_feature(df, col1, col2, operation, new_col_name):
    df = df.copy()

    if col1 not in df.columns or col2 not in df.columns:
        st.warning("Selected columns not found in DataFrame.")
        return df

    if operation == 'multiply':
        df[new_col_name] = df[col1] * df[col2]
    elif operation == 'add':
        df[new_col_name] = df[col1] + df[col2]
    elif operation == 'subtract':
        df[new_col_name] = df[col1] - df[col2]
    elif operation == 'divide':
        df[new_col_name] = df[col1] / df[col2].replace(0, np.nan)
    else:
        st.warning("Invalid operation.")
        return df

    st.success(f"New feature '{new_col_name}' created using {col1} and {col2}.")
    return df


def detect_low_variance_features(df, threshold=0.0):
    numeric_df = df.select_dtypes(include=['number'])
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)

    low_var_cols = numeric_df.columns[~selector.get_support()].tolist()
    return low_var_cols


def remove_low_variance_features(df, columns):
    df = df.drop(columns=columns)
    st.success(f"Removed {len(columns)} low-variance features.")
    return df

def detect_highly_correlated(df, threshold=0.9):
    corr_matrix = df.select_dtypes(include=['number']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop, corr_matrix

def remove_correlated_features(df, columns):
    df = df.drop(columns=columns)
    st.success(f"Removed {len(columns)} highly correlated features.")
    return df

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_feature_importance(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Determine task type
    if y.nunique() <= 10 and y.dtype in ['int', 'object', 'category']:
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    X = pd.get_dummies(X)  # one-hot for non-numeric cols
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False)


def check_class_imbalance(df, target_col):
    class_counts = df[target_col].value_counts()
    return class_counts


from sklearn.model_selection import train_test_split

def split_dataset(df, target_col, test_size=0.2, train_size=0.8, stratify=False):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if stratify and y.nunique() <= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,train_size=train_size, stratify=y, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,train_size=train_size, random_state=42
        )

    return X_train, X_test, y_train, y_test
