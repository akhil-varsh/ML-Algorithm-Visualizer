import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

class MLAlgorithmVisualizer:
    def __init__(self):
        self.classification_algorithms_dict={
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine": SVC(random_state=42, probability=True),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        self.regression_algorithms_dict={
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=42),
            "Lasso Regression": Lasso(random_state=42),
            "Elastic Net": ElasticNet(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
            "Support Vector Regression": SVR(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100)
        }
        self.algorithms={**self.classification_algorithms_dict, **self.regression_algorithms_dict}
        self.regression_algorithms=list(self.regression_algorithms_dict.keys())
        self.classification_algorithms=list(self.classification_algorithms_dict.keys())
        self.scaling_algorithms = [
            "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", 
            "Support Vector Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"
        ]
    
    def determine_problem_type(self, y):
        """Determine if it's a regression or classification problem"""
        unique_values = len(np.unique(y))
        
        # Check if target is categorical/string type
        if y.dtype in ['object', 'category', 'bool']:
            return "classification"
        
        # Check if target has few unique values (likely classification)
        if unique_values <= 20 and y.dtype in ['int64', 'int32', 'int16', 'int8']:
            return "classification"
        
        # Check if all values are integers and within a small range
        if np.all(y == y.astype(int)) and unique_values <= 10:
            return "classification"
        
    
        return "regression"
    
    def train_model(self, algorithm_name, X_train, X_test, y_train, y_test, problem_type):
        """Train and evaluate a model"""
        # Get the appropriate model based on problem type
        if problem_type == "classification":
            model = self.classification_algorithms_dict[algorithm_name]
        else:
            model = self.regression_algorithms_dict[algorithm_name]
        
        # Scale features for algorithms that need it
        if algorithm_name in self.scaling_algorithms:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            scaler = None
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results = {
            "model": model,
            "scaler": scaler,
            "y_pred": y_pred,
            "y_test": y_test
        }
        
        if problem_type == "classification":
            results["accuracy"] = accuracy_score(y_test, y_pred)
            results["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
            results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            results["cv_score"] = cv_scores.mean()
            results["cv_std"] = cv_scores.std()
            
        else:  # regression
            results["r2_score"] = r2_score(y_test, y_pred)
            results["mse"] = mean_squared_error(y_test, y_pred)
            results["rmse"] = np.sqrt(results["mse"])
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            results["cv_score"] = cv_scores.mean()
            results["cv_std"] = cv_scores.std()
        
        return results
    
    def create_confusion_matrix_plot(self, cm, class_names=None):
        """Create confusion matrix heatmap"""
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=class_names,
            y=class_names,
            color_continuous_scale="Blues"
        )
        
        fig.update_layout(
            width=500,
            height=500,
            title_x=0.5
        )
        
        return fig
    
    def create_decision_boundary_plot(self, X, y, model, scaler=None, feature_names=None):
        """Create decision boundary visualization using PCA for dimensionality reduction"""
        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create a mesh
        h = 0.02
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Transform mesh points back to original space and then to model space
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Inverse transform PCA to get back to original feature space
        mesh_original = pca.inverse_transform(mesh_points)
        
        # Apply scaler if used
        if scaler is not None:
            mesh_scaled = scaler.transform(mesh_original)
        else:
            mesh_scaled = mesh_original
        
        # Predict on mesh
        try:
            Z = model.predict(mesh_scaled)
            Z = Z.reshape(xx.shape)
            
            # Create the plot
            fig = go.Figure()
            
            # Add decision boundary
            fig.add_trace(go.Contour(
                x=np.arange(x_min, x_max, h),
                y=np.arange(y_min, y_max, h),
                z=Z,
                showscale=False,
                opacity=0.3,
                colorscale='Viridis'
            ))
            
            # Add data points
            unique_classes = np.unique(y)
            colors = px.colors.qualitative.Set1[:len(unique_classes)]
            
            for i, class_val in enumerate(unique_classes):
                mask = y == class_val
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Class {class_val}',
                    marker=dict(color=colors[i], size=8)
                ))
            
            fig.update_layout(
                title="Decision Boundary (PCA Projection)",
                xaxis_title="First Principal Component",
                yaxis_title="Second Principal Component",
                width=700,
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Could not create decision boundary plot: {str(e)}")
            return None
    
    def create_feature_importance_plot(self, model, feature_names):
        """Create feature importance plot for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance",
                labels={'importance': 'Importance Score', 'feature': 'Features'}
            )
            
            fig.update_layout(
                width=600,
                height=400,
                title_x=0.5
            )
            
            return fig
        
        return None
    
    def create_learning_curve(self, algorithm_name, X, y, problem_type):
        """Create learning curve to show model performance vs training size"""
        from sklearn.model_selection import learning_curve
        
        # Get the appropriate model based on problem type
        if problem_type == "classification":
            model = self.classification_algorithms_dict[algorithm_name]
        else:
            model = self.regression_algorithms_dict[algorithm_name]
        
        # Scale features if needed
        if algorithm_name in self.scaling_algorithms:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Generate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy' if problem_type == 'classification' else 'r2'
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create plot
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            error_y=dict(type='data', array=train_std, visible=True)
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red'),
            error_y=dict(type='data', array=val_std, visible=True)
        ))
        
        fig.update_layout(
            title=f"Learning Curve - {algorithm_name}",
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            width=700,
            height=400,
            title_x=0.5
        )
        
        return fig
