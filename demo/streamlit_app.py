"""Streamlit demo for Trust Calibration in AI Systems.

This interactive demo allows users to explore calibration methods,
uncertainty quantification, and model comparison through a web interface.

WARNING: This project is for research and educational purposes only.
XAI outputs may be unstable or misleading. DO NOT USE FOR REGULATED DECISIONS WITHOUT HUMAN REVIEW.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.datasets import create_synthetic_dataset
from src.eval.metrics import CalibrationEvaluator
from src.models.calibration import CalibratedModel
from src.utils.device import set_seed


# Page configuration
st.set_page_config(
    page_title="Trust Calibration in AI Systems",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔬 Trust Calibration in AI Systems</h1>', unsafe_allow_html=True)

# Warning disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p><strong>This demo is for research and educational purposes only.</strong></p>
    <p>XAI outputs may be unstable or misleading. <strong>DO NOT USE FOR REGULATED DECISIONS WITHOUT HUMAN REVIEW.</strong></p>
    <p>Calibration methods may not generalize across different datasets. Always validate AI system outputs with domain experts.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Dataset selection
dataset_options = {
    "Iris": "iris",
    "Wine": "wine", 
    "Breast Cancer": "breast_cancer",
    "Synthetic": "synthetic"
}

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    options=list(dataset_options.keys()),
    index=0
)

# Model parameters
st.sidebar.subheader("Model Parameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
random_seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)

# Calibration methods
st.sidebar.subheader("Calibration Methods")
use_platt = st.sidebar.checkbox("Platt Scaling", value=True)
use_isotonic = st.sidebar.checkbox("Isotonic Regression", value=True)
use_baseline = st.sidebar.checkbox("Random Forest (Baseline)", value=True)
use_logistic = st.sidebar.checkbox("Logistic Regression", value=True)

# Load dataset
@st.cache_data
def load_dataset(dataset_name: str):
    """Load dataset with caching."""
    set_seed(random_seed)
    
    if dataset_name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == "synthetic":
        X, y, metadata = create_synthetic_dataset(n_samples=1000, n_features=10, n_classes=2)
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        target_names = ["Class 0", "Class 1"]
    
    return X, y, feature_names, target_names

# Load data
X, y, feature_names, target_names = load_dataset(dataset_options[selected_dataset])

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
@st.cache_data
def train_models(X_train, y_train, use_platt, use_isotonic, use_baseline, use_logistic):
    """Train models with caching."""
    models = {}
    
    if use_baseline:
        from sklearn.ensemble import RandomForestClassifier
        models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=random_seed)
        models["Random Forest"].fit(X_train, y_train)
    
    if use_platt:
        models["Platt Scaling"] = CalibratedModel(
            calibration_method="platt"
        )
        models["Platt Scaling"].fit(X_train, y_train)
    
    if use_isotonic:
        models["Isotonic Regression"] = CalibratedModel(
            calibration_method="isotonic"
        )
        models["Isotonic Regression"].fit(X_train, y_train)
    
    if use_logistic:
        from sklearn.linear_model import LogisticRegression
        models["Logistic Regression"] = LogisticRegression(random_state=random_seed)
        models["Logistic Regression"].fit(X_train, y_train)
    
    return models

models = train_models(X_train_scaled, y_train, use_platt, use_isotonic, use_baseline, use_logistic)

# Evaluate models
evaluator = CalibrationEvaluator()
results = evaluator.evaluate_multiple_models(models, X_test_scaled, y_test)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Calibration Analysis", "🎯 Model Comparison", "🔍 Detailed Results"])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset", selected_dataset)
        st.metric("Samples", len(X))
        st.metric("Features", X.shape[1])
    
    with col2:
        st.metric("Classes", len(np.unique(y)))
        st.metric("Training Samples", len(X_train))
        st.metric("Test Samples", len(X_test))
    
    with col3:
        st.metric("Models Trained", len(models))
        st.metric("Random Seed", random_seed)
        st.metric("Test Size", f"{test_size:.1%}")
    
    # Dataset preview
    st.subheader("Dataset Preview")
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    st.dataframe(df.head(10), use_container_width=True)
    
    # Class distribution
    st.subheader("Class Distribution")
    class_counts = pd.Series(y).value_counts().sort_index()
    fig = px.bar(
        x=[target_names[i] for i in class_counts.index],
        y=class_counts.values,
        title="Class Distribution",
        labels={'x': 'Class', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Calibration Analysis")
    
    # Calibration curves
    st.subheader("Calibration Curves")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(models.keys()),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, (name, model) in enumerate(models.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        
        y_prob = model.predict_proba(X_test_scaled)
        if y_prob.shape[1] > 2:
            y_prob_positive = np.max(y_prob, axis=1)
        else:
            y_prob_positive = y_prob[:, 1]
        
        # Calculate calibration curve
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob_positive, n_bins=10)
        
        fig.add_trace(
            go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers',
                      name=f'{name} Calibration', line=dict(width=3)),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      name='Perfect Calibration', line=dict(dash='dash', color='gray')),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Mean Predicted Probability")
    fig.update_yaxes(title_text="Fraction of Positives")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Reliability diagrams
    st.subheader("Reliability Diagrams")
    
    for name, model in models.items():
        st.write(f"**{name}**")
        
        y_prob = model.predict_proba(X_test_scaled)
        if y_prob.shape[1] > 2:
            y_prob_positive = np.max(y_prob, axis=1)
        else:
            y_prob_positive = y_prob[:, 1]
        
        # Calculate bin data
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob_positive > bin_lower) & (y_prob_positive <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(y_test[in_bin].mean())
                bin_counts.append(prop_in_bin)
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=bin_centers, y=bin_accuracies, mode='lines+markers',
                      name='Calibration', line=dict(width=3)),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      name='Perfect Calibration', line=dict(dash='dash', color='gray')),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(x=bin_centers, y=bin_counts, name='Sample Count', opacity=0.3),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Mean Predicted Probability")
        fig.update_yaxes(title_text="Fraction of Positives", secondary_y=False)
        fig.update_yaxes(title_text="Number of Samples", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Model Comparison")
    
    # Metrics comparison
    st.subheader("Performance Metrics")
    
    metrics_df = pd.DataFrame(results).T
    st.dataframe(metrics_df, use_container_width=True)
    
    # Metrics visualization
    st.subheader("Metrics Comparison")
    
    metrics_to_plot = st.multiselect(
        "Select metrics to plot",
        options=list(metrics_df.columns),
        default=list(metrics_df.columns)[:3]
    )
    
    if metrics_to_plot:
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=list(metrics_df.index),
                y=metrics_df[metric]
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Metric Value",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Leaderboard
    st.subheader("Model Rankings")
    
    for metric in metrics_df.columns:
        st.write(f"**{metric.upper()}** (Lower is better)" if metric not in ["accuracy", "roc_auc"] else f"**{metric.upper()}** (Higher is better)")
        
        if metric in ["accuracy", "roc_auc"]:
            ranking = metrics_df[metric].sort_values(ascending=False)
        else:
            ranking = metrics_df[metric].sort_values(ascending=True)
        
        for i, (model, value) in enumerate(ranking.items(), 1):
            st.write(f"{i}. {model}: {value:.4f}")

with tab4:
    st.header("Detailed Results")
    
    # Model details
    for name, model in models.items():
        with st.expander(f"**{name}** Details"):
            st.write(f"**Type:** {type(model).__name__}")
            st.write(f"**Calibration Method:** {getattr(model, 'calibration_method', 'N/A')}")
            
            # Performance metrics
            metrics = results[name]
            st.write("**Performance Metrics:**")
            for metric, value in metrics.items():
                st.write(f"- {metric}: {value:.4f}")
            
            # Predictions sample
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
            
            st.write("**Sample Predictions:**")
            sample_df = pd.DataFrame({
                'True Label': y_test[:10],
                'Predicted Label': y_pred[:10],
                'Max Probability': np.max(y_prob[:10], axis=1)
            })
            st.dataframe(sample_df, use_container_width=True)
    
    # Download results
    st.subheader("Download Results")
    
    # Convert results to CSV
    results_df = pd.DataFrame(results).T
    csv = results_df.to_csv()
    
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"calibration_results_{selected_dataset.lower()}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Trust Calibration in AI Systems - Research & Educational Use Only</p>
    <p>⚠️ DO NOT USE FOR REGULATED DECISIONS WITHOUT HUMAN REVIEW ⚠️</p>
</div>
""", unsafe_allow_html=True)
