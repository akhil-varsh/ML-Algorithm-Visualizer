import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_sample_datasets():
    datasets = {
        "Iris": load_iris(as_frame=True).frame,
        "Wine": load_wine(as_frame=True).frame,
        "Breast Cancer": load_breast_cancer(as_frame=True).frame
    }
    return datasets

def preprocess_data(df, target_column):
    """Preprocess the dataset for ML algorithms"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=[np.number]).columns
    
    # Encode categorical variables
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Encode target if categorical
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    
    return X, y, le_dict, target_encoder, categorical_columns, numerical_columns

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    # Only include numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.empty:
        return None
    
    corr_matrix = numerical_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5
    )
    
    return fig

def create_distribution_plots(df, target_column=None):
    """Create distribution plots for numerical features"""
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    if target_column and target_column in numerical_columns:
        numerical_columns = numerical_columns.drop(target_column)
    
    if len(numerical_columns) == 0:
        return None
    
    # Create subplots
    n_cols = min(3, len(numerical_columns))
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(numerical_columns),
        vertical_spacing=0.1
    )
    
    for i, col in enumerate(numerical_columns):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row,
            col=col_pos
        )
    
    fig.update_layout(
        title="Feature Distributions",
        height=300 * n_rows,
        showlegend=False
    )
    
    return fig

def create_pairplot(df, target_column=None, max_features=5):
    """Create pairplot for feature relationships"""
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    if target_column and target_column in numerical_columns:
        numerical_columns = numerical_columns.drop(target_column)
    
    # Limit to max_features to avoid overcrowding
    if len(numerical_columns) > max_features:
        numerical_columns = numerical_columns[:max_features]
    
    if len(numerical_columns) < 2:
        return None
    
    # Create scatter matrix
    fig = px.scatter_matrix(
        df,
        dimensions=list(numerical_columns),
        color=target_column if target_column else None,
        title="Feature Relationships (Pairplot)"
    )
    
    fig.update_layout(
        width=800,
        height=800,
        title_x=0.5
    )
    
    return fig

def get_dataset_summary(df):
    """Get comprehensive dataset summary"""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numerical_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=['object']).columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "duplicate_rows": df.duplicated().sum()
    }
    
    # Basic statistics for numerical columns
    if summary["numerical_columns"]:
        summary["statistics"] = df[summary["numerical_columns"]].describe().to_dict()
    
    return summary
