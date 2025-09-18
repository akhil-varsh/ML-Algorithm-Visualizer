# ML Algorithm Visualizer with Groq AI

A comprehensive machine learning visualization tool that lets you explore datasets, train multiple algorithms, and get AI-powered insights through Groq integration.

## What it does

This app makes machine learning accessible by providing an intuitive interface to:
- Upload and explore your datasets with interactive visualizations
- Train and compare 9+ different ML algorithms (both classification and regression)
- Get AI-powered explanations and recommendations for your models
- Visualize model performance with charts, confusion matrices, and learning curves

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                       │
│                         (app.py)                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   UI Components                                 │
│                 (ui_components.py)                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │    Data     │ │     ML      │ │   Model     │ │    AI     │ │
│  │Visualization│ │ Algorithms  │ │ Comparison  │ │ Assistant │ │
│  │     Tab     │ │     Tab     │ │     Tab     │ │    Tab    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────┬───────────────┬───────────────┬───────────────┬───────────┘
      │               │               │               │
┌─────▼─────┐ ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼─────────┐
│   Data    │ │      ML      │ │   Session   │ │     Groq      │
│  Handler  │ │  Algorithms  │ │    State    │ │   Assistant   │
│           │ │              │ │   Manager   │ │               │
│• Dataset  │ │• 9 Regression│ │• Model      │ │• Chat         │
│  Loading  │ │  Algorithms  │ │  Results    │ │• Analysis     │
│• Preprocessing│ │• 6 Classification│ │• Dataset    │ │• Recommendations│
│• Visualization│ │  Algorithms  │ │  Storage    │ │• Explanations │
│• Statistics│ │• Training    │ │• UI State   │ │               │
│           │ │• Evaluation  │ │             │ │               │
└───────────┘ └──────────────┘ └─────────────┘ └───────────────┘
      │               │                               │
┌─────▼─────┐ ┌───────▼──────┐                 ┌─────▼─────────┐
│  Plotly   │ │   Scikit-    │                 │   Groq API    │
│  Charts   │ │    Learn     │                 │   (External)  │
│           │ │              │                 │               │
│• Correlation│ │• Models     │                 │• gemma2-9b-it │
│• Distribution│ │• Metrics    │                 │• Chat         │
│• Pairplots │ │• Validation │                 │• Analysis     │
│• Heatmaps  │ │• Visualization│                │               │
└───────────┘ └──────────────┘                 └───────────────┘
```

## Core Components

### Data Handler (`data_handler.py`)
Manages all data operations including loading sample datasets (Iris, Wine, Breast Cancer), preprocessing features, handling categorical variables, and creating visualizations like correlation heatmaps and distribution plots.

### ML Algorithms (`ml_algorithms.py`)
Contains the MLAlgorithmVisualizer class that handles:
- **Regression**: Linear, Ridge, Lasso, Elastic Net, KNN, SVR, Decision Tree, Random Forest, Gradient Boosting
- **Classification**: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting
- Model training, evaluation, and visualization (learning curves, feature importance, decision boundaries)

### Groq AI Assistant (`groq_assistant.py`)
Integrates with Groq's API to provide:
- Dataset analysis and insights
- Algorithm recommendations based on your data
- Model results explanations in plain English
- Interactive chat for ML questions

### UI Components (`ui_components.py`)
Four main tabs that provide the user interface:
1. **Data & Visualization**: Upload datasets, explore data with interactive charts
2. **ML Algorithms**: Configure and train models with real-time results
3. **Model Comparison**: Compare performance across different algorithms
4. **AI Assistant**: Get AI-powered insights and recommendations

## Getting Started

1. **Install dependencies**:
```bash
pip install streamlit pandas plotly scikit-learn groq python-dotenv
```

2. **Set up Groq API** (optional but recommended):
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Or add it to Streamlit secrets in `.streamlit/secrets.toml`:
```
GROQ_API_KEY = "your_groq_api_key_here"
```

3. **Run the app**:
```bash
streamlit run app.py
```

## Features

### Data Exploration
- Upload CSV files or use built-in sample datasets
- Interactive correlation heatmaps and distribution plots
- Dataset statistics and missing value analysis
- Pairplot visualizations with target coloring

### Machine Learning
- Automatic problem type detection (classification vs regression)
- 15 different algorithms to choose from
- Cross-validation scoring and performance metrics
- Visual model evaluation (confusion matrices, prediction plots)
- Feature importance analysis for tree-based models
- Learning curves to understand model behavior

### AI Integration
- Chat with AI about your data and models
- Get personalized algorithm recommendations
- Understand model results with AI explanations
- Compare multiple models with AI insights

## Sample Datasets

The app includes three classic ML datasets:
- **Iris**: Flower classification (4 features, 3 classes)
- **Wine**: Wine quality classification (13 features, 3 classes)  
- **Breast Cancer**: Cancer diagnosis (30 features, 2 classes)

## Technical Details

Built with:
- **Streamlit** for the web interface
- **Scikit-learn** for machine learning algorithms
- **Plotly** for interactive visualizations
- **Groq API** for AI-powered insights
- **Pandas** for data manipulation

The app uses session state to maintain data and model results across tabs, allowing you to train multiple models and compare them seamlessly.

## Tips

- Start with the sample datasets to get familiar with the interface
- Try different algorithms on the same dataset to see which performs best
- Use the AI assistant to understand why certain algorithms work better for your data
- The model comparison tab helps you make informed decisions about which algorithm to use

Perfect for data scientists, students, and anyone wanting to understand machine learning through hands-on experimentation.