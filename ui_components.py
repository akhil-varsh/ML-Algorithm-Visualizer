import streamlit as st
import pandas as pd
import plotly.express as px
from data_handler import load_sample_datasets, preprocess_data, create_correlation_heatmap, create_distribution_plots, create_pairplot
from ml_algorithms import MLAlgorithmVisualizer
from groq_assistant import GroqAIAssistant

def data_visualization_tab():
    st.header("📊 Data Upload & Exploration")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your dataset in CSV format")
    with col2:
        st.subheader("Or Use Sample Data")
        sample_datasets = load_sample_datasets()
        selected_sample = st.selectbox("Choose a sample dataset", ["None"] + list(sample_datasets.keys()), help="Select a built-in dataset for quick exploration")
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            st.session_state.dataset = df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    elif selected_sample != "None":
        df = sample_datasets[selected_sample].copy()
        st.success(f"✅ {selected_sample} dataset loaded! Shape: {df.shape}")
        st.session_state.dataset = df
    elif st.session_state.dataset is not None:
        df = st.session_state.dataset
    if df is not None:
        st.subheader("📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), width='stretch')
        summary = get_dataset_summary(df)
        with st.expander("📊 Detailed Dataset Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Information:**")
                info_df = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str), 'Non-Null Count': df.count(), 'Null Count': df.isnull().sum()})
                st.dataframe(info_df, width='stretch')
            with col2:
                if summary["numerical_columns"]:
                    st.write("**Numerical Statistics:**")
                    st.dataframe(df[summary["numerical_columns"]].describe(), width='stretch')
        st.subheader("📈 Data Visualization")
        viz_tabs = st.tabs(["Correlation", "Distributions", "Relationships"])
        with viz_tabs[0]:
            st.write("**Feature Correlation Heatmap**")
            corr_fig = create_correlation_heatmap(df)
            if corr_fig:
                st.plotly_chart(corr_fig, width='stretch')
            else:
                st.info("No numerical features available for correlation analysis.")
        with viz_tabs[1]:
            st.write("**Feature Distributions**")
            dist_fig = create_distribution_plots(df)
            if dist_fig:
                st.plotly_chart(dist_fig, width='stretch')
            else:
                st.info("No numerical features available for distribution plots.")
        with viz_tabs[2]:
            st.write("**Feature Relationships (Pairplot)**")
            target_col = st.selectbox("Select target column for coloring (optional)", ["None"] + list(df.columns), key="pairplot_target")
            target_col = None if target_col == "None" else target_col
            pair_fig = create_pairplot(df, target_col)
            if pair_fig:
                st.plotly_chart(pair_fig, width='stretch')
            else:
                st.info("Need at least 2 numerical features for pairplot.")
    else:
        st.info("👆 Please upload a dataset or select a sample dataset to begin exploration.")


def ml_algorithms_tab():
    st.header("🔬 Machine Learning Algorithms")
    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first in the 'Data & Visualization' tab.")
        return
    df = st.session_state.dataset
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Configuration")
        target_column = st.selectbox("Select Target Column", df.columns, help="Choose the column you want to predict")
        available_features = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect("Select Features", available_features, default=available_features[:min(10, len(available_features))], help="Choose features to use for prediction")
        if not selected_features:
            st.error("Please select at least one feature.")
            return
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Proportion of data to use for testing")
    with col2:
        if selected_features:
            feature_df = df[selected_features + [target_column]]
            X, y, le_dict, target_encoder, cat_cols, num_cols = preprocess_data(feature_df, target_column)
            ml_viz = MLAlgorithmVisualizer()
            problem_type = ml_viz.determine_problem_type(y)
            st.subheader(f"🎯 Problem Type: {problem_type.title()}")
            if problem_type == "classification":
                available_algorithms = ml_viz.classification_algorithms
            else:
                available_algorithms = ml_viz.regression_algorithms
            selected_algorithm = st.selectbox("Select Algorithm", available_algorithms, help="Choose the machine learning algorithm to use")
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if problem_type == "classification" else None)
                    results = ml_viz.train_model(selected_algorithm, X_train, X_test, y_train, y_test, problem_type)
                    if 'model_results' not in st.session_state:
                        st.session_state.model_results = {}
                    st.session_state.model_results[selected_algorithm] = results
                    st.success("✅ Model trained successfully!")
                    st.subheader("📊 Model Performance")
                    if problem_type == "classification":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{results['accuracy']:.3f}")
                        with col2:
                            st.metric("CV Score", f"{results['cv_score']:.3f}")
                        with col3:
                            st.metric("CV Std", f"{results['cv_std']:.3f}")
                        st.subheader("🔍 Confusion Matrix")
                        cm_fig = ml_viz.create_confusion_matrix_plot(results['confusion_matrix'])
                        st.plotly_chart(cm_fig, width='stretch')
                        with st.expander("📋 Detailed Classification Report"):
                            report_df = pd.DataFrame(results['classification_report']).transpose()
                            st.dataframe(report_df, width='stretch')
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{results['r2_score']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{results['rmse']:.3f}")
                        with col3:
                            st.metric("CV Score", f"{results['cv_score']:.3f}")
                        with col4:
                            st.metric("CV Std", f"{results['cv_std']:.3f}")
                        st.subheader("📈 Predictions vs Actual")
                        pred_fig = px.scatter(x=results['y_test'], y=results['y_pred'], labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title="Predictions vs Actual Values")
                        pred_fig.add_shape(type="line", line=dict(dash="dash"), x0=results['y_test'].min(), y0=results['y_test'].min(), x1=results['y_test'].max(), y1=results['y_test'].max())
                        st.plotly_chart(pred_fig, width='stretch')
                    st.subheader("🎨 Model Visualizations")
                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        if problem_type == "classification" and len(selected_features) >= 2:
                            st.write("**Decision Boundary**")
                            boundary_fig = ml_viz.create_decision_boundary_plot(X_test, y_test, results['model'], results['scaler'], selected_features)
                            if boundary_fig:
                                st.plotly_chart(boundary_fig, width='stretch')
                    with viz_col2:
                        st.write("**Feature Importance**")
                        importance_fig = ml_viz.create_feature_importance_plot(results['model'], selected_features)
                        if importance_fig:
                            st.plotly_chart(importance_fig, width='stretch')
                        else:
                            st.info("Feature importance not available for this algorithm.")
                    st.subheader("📚 Learning Curve")
                    learning_fig = ml_viz.create_learning_curve(selected_algorithm, X, y, problem_type)
                    st.plotly_chart(learning_fig, width='stretch')


def model_comparison_tab():
    st.header("📈 Model Comparison")
    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first in the 'Data & Visualization' tab.")
        return
    if 'model_results' not in st.session_state or not st.session_state.model_results:
        st.info("🔬 Train some models in the 'ML Algorithms' tab first to compare them here.")
        return
    df = st.session_state.dataset
    results = st.session_state.model_results
    st.subheader("🏆 Performance Comparison")
    comparison_data = []
    problem_type = None
    for algo_name, result in results.items():
        if 'accuracy' in result:
            problem_type = "classification"
            comparison_data.append({'Algorithm': algo_name, 'Accuracy': result['accuracy'], 'CV Score': result['cv_score'], 'CV Std': result['cv_std']})
        else:
            problem_type = "regression"
            comparison_data.append({'Algorithm': algo_name, 'R² Score': result['r2_score'], 'RMSE': result['rmse'], 'CV Score': result['cv_score'], 'CV Std': result['cv_std']})
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, width='stretch')
        col1, col2 = st.columns(2)
        with col1:
            if problem_type == "classification":
                fig = px.bar(comparison_df, x='Algorithm', y='Accuracy', title="Model Accuracy Comparison", text='Accuracy')
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            else:
                fig = px.bar(comparison_df, x='Algorithm', y='R² Score', title="Model R² Score Comparison", text='R² Score')
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = px.bar(comparison_df, x='Algorithm', y='CV Score', title="Cross-Validation Score Comparison", text='CV Score')
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, width='stretch')
        if problem_type == "classification":
            best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            metric_name = "Accuracy"
            metric_value = best_model['Accuracy']
        else:
            best_model = comparison_df.loc[comparison_df['R² Score'].idxmax()]
            metric_name = "R² Score"
            metric_value = best_model['R² Score']
        st.success(f"🏆 **Best Model:** {best_model['Algorithm']} with {metric_name}: {metric_value:.3f}")


def ai_assistant_tab():
    st.header("🤖 AI Assistant")
    ai_assistant = GroqAIAssistant()
    if not ai_assistant.is_available():
        st.error("❌ Groq AI Assistant is not available. Please check your API key configuration.")
        st.info("💡 Make sure to set your GROQ_API_KEY in the .env file or Streamlit secrets.")
        return
    st.success("✅ AI Assistant is ready to help!")
    assistant_tabs = st.tabs(["💬 Chat", "🔍 Dataset Analysis", "🎯 Algorithm Recommendations", "📊 Results Explanation"])
    with assistant_tabs[0]:
        st.subheader("💬 Chat with AI Assistant")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)
        user_question = st.chat_input("Ask me anything about machine learning or your data...")
        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            st.chat_message("user").write(user_question)
            context = {}
            if st.session_state.dataset is not None:
                context["dataset_shape"] = st.session_state.dataset.shape
                context["dataset_columns"] = list(st.session_state.dataset.columns)
            if 'model_results' in st.session_state:
                context["trained_models"] = list(st.session_state.model_results.keys())
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ai_assistant.answer_question(user_question, context)
                    st.write(response)
                    st.session_state.chat_history.append(("assistant", response))
    with assistant_tabs[1]:
        st.subheader("🔍 Dataset Analysis")
        if st.session_state.dataset is not None:
            if st.button("🔍 Analyze My Dataset", type="primary"):
                with st.spinner("Analyzing your dataset..."):
                    analysis = ai_assistant.analyze_dataset(st.session_state.dataset)
                    st.markdown(analysis)
        else:
            st.info("📊 Load a dataset first to get AI-powered analysis.")
    with assistant_tabs[2]:
        st.subheader("🎯 Algorithm Recommendations")
        if st.session_state.dataset is not None:
            target_col = st.selectbox("Select target column for recommendations", st.session_state.dataset.columns, key="ai_target_col")
            if st.button("🎯 Get Algorithm Recommendations", type="primary"):
                with st.spinner("Analyzing and recommending algorithms..."):
                    ml_viz = MLAlgorithmVisualizer()
                    y = st.session_state.dataset[target_col]
                    problem_type = ml_viz.determine_problem_type(y)
                    recommendations = ai_assistant.recommend_algorithms(st.session_state.dataset, target_col, problem_type)
                    st.markdown(recommendations)
        else:
            st.info("📊 Load a dataset first to get algorithm recommendations.")
    with assistant_tabs[3]:
        st.subheader("📊 Results Explanation")
        if 'model_results' in st.session_state and st.session_state.model_results:
            selected_model = st.selectbox("Select model to explain", list(st.session_state.model_results.keys()), key="ai_model_explain")
            if st.button("📊 Explain Results", type="primary"):
                with st.spinner("Explaining model results..."):
                    results = st.session_state.model_results[selected_model]
                    problem_type = "classification" if 'accuracy' in results else "regression"
                    explanation = ai_assistant.explain_results(selected_model, results, problem_type)
                    st.markdown(explanation)
            if len(st.session_state.model_results) > 1:
                if st.button("🏆 Compare All Models", type="secondary"):
                    with st.spinner("Comparing all trained models..."):
                        first_result = list(st.session_state.model_results.values())[0]
                        problem_type = "classification" if 'accuracy' in first_result else "regression"
                        comparison = ai_assistant.compare_algorithms(st.session_state.model_results, problem_type)
                        st.markdown(comparison)
        else:
            st.info("🔬 Train some models first to get AI explanations of the results.")
   
