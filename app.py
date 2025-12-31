import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
import os
from io import StringIO
from ui_components import data_visualization_tab, ml_algorithms_tab, model_comparison_tab, ai_assistant_tab
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(
    page_title="ML Algorithm Visualizer with Groq AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🤖 ML Algorithm Visualizer with Groq AI</h1>', unsafe_allow_html=True)
    st.markdown("Interactive machine learning visualization with intelligent AI assistance")
    
    if 'dataset' not in st.session_state:
        st.session_state.dataset=None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    
    tab1, tab2, tab3, tab4=st.tabs(["📊 Data & Visualization", "🔬 ML Algorithms", "📈 Model Comparison", "🤖 AI Assistant"])
    
    with tab1:
        data_visualization_tab()
    with tab2:
        ml_algorithms_tab()
    with tab3:
        model_comparison_tab()
    with tab4:
        ai_assistant_tab()

if __name__=="__main__":
    main()
