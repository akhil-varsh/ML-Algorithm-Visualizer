import streamlit as st
from groq import Groq
import pandas as pd
import json
import os
from typing import Dict, List, Any

class GroqAIAssistant:
    def __init__(self):
        self.client=None
        self.model="gemma2-9b-it"
        self.initialize_client()
    
    def initialize_client(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GROQ_API_KEY"]
            except:
                pass
        if api_key:
            self.client=Groq(api_key=api_key)
    
    def is_available(self):
        return self.client is not None
    
    def analyze_dataset(self, df: pd.DataFrame, target_column: str=None) -> str:
        if not self.is_available():
            return "Groq AI assistant is not available. Please configure your API key."
        summary=self._create_dataset_summary(df, target_column)
        
        prompt = f"""
        As an expert data scientist and ML educator, analyze this dataset and provide educational insights:

        Dataset Summary:
        {summary}

        Please provide:
        1. Key insights about the data structure and quality
        2. Potential data preprocessing steps needed
        3. Suitable machine learning algorithms for this dataset
        4. Expected challenges and how to address them
        5. Educational tips for students learning from this dataset

        Keep your response educational, practical, and encouraging for learners.
        """
        
        return self._get_completion(prompt)
    
    def recommend_algorithms(self, df: pd.DataFrame, target_column: str, problem_type: str) -> str:
        if not self.is_available():
            return "Groq AI assistant is not available. Please configure your API key."
        dataset_info=self._create_dataset_summary(df, target_column)
        
        prompt = f"""
        As an ML expert, recommend the best algorithms for this dataset:

        Dataset Information:
        {dataset_info}

        Problem Type: {problem_type}

        Please recommend:
        1. Top 3 most suitable algorithms and why
        2. Algorithm parameters to consider tuning
        3. Expected performance characteristics
        4. Pros and cons of each recommended algorithm
        5. Learning objectives for students using these algorithms

        Focus on educational value and practical implementation tips.
        """
        
        return self._get_completion(prompt)
    
    def explain_results(self, algorithm_name: str, results: Dict, problem_type: str) -> str:
        if not self.is_available():
            return "Groq AI assistant is not available. Please configure your API key."
        
        # Format results for the prompt
        results_summary = self._format_results(results, problem_type)
        
        prompt = f"""
        As an ML educator, explain these {algorithm_name} results to help students understand:

        Algorithm: {algorithm_name}
        Problem Type: {problem_type}
        Results: {results_summary}

        Please explain:
        1. What these metrics mean in simple terms
        2. How to interpret the performance (good/bad/average)
        3. What the results tell us about the model's behavior
        4. Suggestions for improvement
        5. Key learning points for students

        Use clear, educational language suitable for ML beginners.
        """
        
        return self._get_completion(prompt)
    
    def answer_question(self, question: str, context: Dict=None) -> str:
        if not self.is_available():
            return "Groq AI assistant is not available. Please configure your API key."
        
        context_str = ""
        if context:
            context_str = f"\nContext about user's current work:\n{json.dumps(context, indent=2)}"
        
        prompt = f"""
        As an expert ML educator and data scientist, answer this question clearly and educationally:

        Question: {question}
        {context_str}

        Provide a comprehensive but accessible answer that:
        1. Directly addresses the question
        2. Includes relevant examples when helpful
        3. Explains concepts in beginner-friendly terms
        4. Offers practical tips or next steps
        5. Encourages further learning

        Keep the tone supportive and educational.
        """
        
        return self._get_completion(prompt)
    
    def compare_algorithms(self, results_dict: Dict, problem_type: str) -> str:
        if not self.is_available():
            return "Groq AI assistant is not available. Please configure your API key."
        
        comparison_data = {}
        for algo_name, results in results_dict.items():
            comparison_data[algo_name] = self._format_results(results, problem_type)
        
        prompt = f"""
        As an ML expert, compare these algorithm results and provide educational insights:

        Problem Type: {problem_type}
        Algorithm Results:
        {json.dumps(comparison_data, indent=2)}

        Please provide:
        1. Performance ranking with explanations
        2. Strengths and weaknesses of each algorithm on this dataset
        3. Which algorithm to choose and why
        4. What these comparisons teach us about algorithm selection
        5. Recommendations for further experimentation

        Focus on helping students understand algorithm selection principles.
        """
        
        return self._get_completion(prompt)
    
    def _create_dataset_summary(self, df: pd.DataFrame, target_column: str=None) -> str:
        summary={
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(df.columns),
            "target_column": target_column,
            "numerical_features": int(len(df.select_dtypes(include=['number']).columns)),
            "categorical_features": int(len(df.select_dtypes(include=['object']).columns)),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        }
        
        if target_column and target_column in df.columns:
            if df[target_column].dtype in ['object', 'category']:
                summary["target_classes"] = int(df[target_column].nunique())
                # Convert value counts to regular Python types
                class_dist = df[target_column].value_counts().to_dict()
                summary["class_distribution"] = {str(k): int(v) for k, v in class_dist.items()}
            else:
                summary["target_range"] = [float(df[target_column].min()), float(df[target_column].max())]
        
        return json.dumps(summary, indent=2)
    
    def _format_results(self, results: Dict, problem_type: str) -> Dict:
        formatted={}
        
        if problem_type == "classification":
            formatted["accuracy"] = results.get("accuracy", 0)
            formatted["cv_score"] = results.get("cv_score", 0)
            formatted["cv_std"] = results.get("cv_std", 0)
        else:
            formatted["r2_score"] = results.get("r2_score", 0)
            formatted["rmse"] = results.get("rmse", 0)
            formatted["cv_score"] = results.get("cv_score", 0)
            formatted["cv_std"] = results.get("cv_std", 0)
        
        return formatted
    
    def _get_completion(self, prompt: str) -> str:
        completion=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    
    def get_streaming_response(self, prompt: str):
        completion=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content