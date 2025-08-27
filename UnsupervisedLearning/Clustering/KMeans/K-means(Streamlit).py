import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
MODEL_FILE = 'models/kmeans_model.pkl'
SCALER_FILE = 'models/scaler.pkl'
FEATURES_FILE = 'models/customer_features.pkl'

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file, delimiter=',', encoding='ISO-8859-1')
        
        # Data cleaning
        df = df.dropna()
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] >= 0]
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['sales'] = df['Quantity'] * df['UnitPrice']
        
        st.success(f"✅ Data loaded successfully: {len(df)} transactions")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def build_customer_features(df, reference_date=None):
    """Build comprehensive customer intelligence features."""
    if reference_date is None:
        reference_date = df['InvoiceDate'].max()
    
    # Aggregation functions
    aggregation = {
        'InvoiceDate': ['count', 'max', 'min'],
        'sales': ['sum', 'mean', 'std'],
        'Quantity': ['sum', 'mean'],
        'InvoiceNo': 'nunique',
        'StockCode': 'nunique',
        'Country': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }
    
    customer_features = df.groupby('CustomerID').agg(aggregation).round(2)
    customer_features.columns = [
        'total_transactions', 'last_purchase', 'first_purchase',
        'total_spent', 'avg_order_value', 'spending_consistency',
        'total_quantity', 'avg_quantity_per_transaction',
        'unique_orders', 'product_diversity', 'primary_country'
    ]
    
    # Calculate advanced features
    customer_features['recency_days'] = (reference_date - customer_features['last_purchase']).dt.days
    customer_features['customer_lifespan_days'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.days
    customer_features['avg_days_between_orders'] = customer_features['customer_lifespan_days'] / (customer_features['unique_orders'] - 1)
    customer_features['avg_days_between_orders'] = customer_features['avg_days_between_orders'].fillna(0)
    
    # Handle spending consistency
    customer_features['spending_consistency'] = customer_features['spending_consistency'].fillna(0)
    customer_features['consistency_score'] = 1 / (1 + customer_features['spending_consistency'] / customer_features['avg_order_value'])
    customer_features['consistency_score'] = customer_features['consistency_score'].fillna(0.5)
    
    # Lifecycle staging
    def determine_lifecycle_stage(recency):
        if recency <= 30:
            return 'Active'
        elif recency <= 90:
            return 'Recent'
        elif recency <= 365:
            return 'Dormant'
        else:
            return 'Lost'
    
    customer_features['lifecycle_stage'] = customer_features['recency_days'].apply(determine_lifecycle_stage)
    
    # Purchase rhythm score
    customer_features['purchase_rhythm_score'] = 1 / (1 + customer_features['avg_days_between_orders'] / 30)
    customer_features['purchase_rhythm_score'] = customer_features['purchase_rhythm_score'].fillna(0)
    
    return customer_features

def find_optimal_clusters(scaled_features, max_k=10):
    """Find optimal number of clusters using silhouette score."""
    silhouette_scores = []
    k_range = range(2, min(max_k + 1, len(scaled_features)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        silhouette_scores.append(score)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k, dict(zip(k_range, silhouette_scores))

def train_clustering_model(customer_features, n_clusters=5):
    """Train the clustering model and save artifacts."""
    feature_cols = [
        'total_spent', 'avg_order_value', 'consistency_score',
        'recency_days', 'total_transactions', 'product_diversity',
        'purchase_rhythm_score'
    ]
    
    # Prepare data for clustering
    clustering_data = customer_features[feature_cols].fillna(customer_features[feature_cols].median())
    
    # Scale features
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(clustering_data)
    
    # Train clustering model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to customer features
    customer_features['cluster'] = cluster_labels
    
    # Save model artifacts
    joblib.dump(kmeans, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(customer_features, FEATURES_FILE)
    
    st.success(f"✅ Model trained and saved with {n_clusters} clusters!")
    
    return customer_features, kmeans, scaler

def load_model_artifacts():
    """Load saved model artifacts."""
    try:
        if all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, FEATURES_FILE]):
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            features = joblib.load(FEATURES_FILE)
            return model, scaler, features
        else:
            return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

def predict_customer_cluster(model, scaler, input_features):
    """Predict cluster for new customer data."""
    try:
        input_array = np.array(input_features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        cluster = model.predict(scaled_input)[0]
        return cluster
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

def generate_cluster_profiles(customer_features):
    """Generate business intelligence profiles for each cluster."""
    profiles = {}
    
    for cluster_id in sorted(customer_features['cluster'].unique()):
        cluster_data = customer_features[customer_features['cluster'] == cluster_id]
        
        profile = {
            'size': len(cluster_data),
            'avg_spent': cluster_data['total_spent'].mean(),
            'avg_recency': cluster_data['recency_days'].mean(),
            'avg_transactions': cluster_data['total_transactions'].mean(),
            'avg_order_value': cluster_data['avg_order_value'].mean(),
            'dominant_lifecycle': cluster_data['lifecycle_stage'].mode()[0],
            'consistency_score': cluster_data['consistency_score'].mean()
        }
        
        # Business segment naming
        if profile['avg_spent'] >= customer_features['total_spent'].quantile(0.8):
            value_tier = "Premium"
        elif profile['avg_spent'] >= customer_features['total_spent'].quantile(0.6):
            value_tier = "High-Value"
        else:
            value_tier = "Standard"
        
        if profile['avg_recency'] <= 30:
            engagement = "Active"
        elif profile['avg_recency'] <= 90:
            engagement = "Recent"
        else:
            engagement = "At-Risk"
        
        profile['segment_name'] = f"{value_tier} {engagement}"
        profiles[f'Cluster {cluster_id}'] = profile
    
    return profiles

def main():
    st.set_page_config(
        page_title="Customer Segmentation Platform",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Enterprise Customer Segmentation Platform")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Data Upload & Training", "Model Prediction", "Business Intelligence"])
    
    if page == "Data Upload & Training":
        st.header("📊 Data Upload & Model Training")
        
        uploaded_file = st.file_uploader(
            "Upload your customer transaction data (CSV)", 
            type=['csv'],
            help="Upload a CSV file with transaction data including CustomerID, Quantity, UnitPrice, InvoiceDate"
        )
        
        if uploaded_file is not None:
            df = load_and_preprocess_data(uploaded_file)
            
            if df is not None:
                # Display data summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    st.metric("Unique Customers", df['CustomerID'].nunique())
                with col3:
                    st.metric("Date Range", f"{(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days")
                with col4:
                    st.metric("Total Revenue", f"${df['sales'].sum():,.2f}")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Build customer features
                with st.spinner("Building customer intelligence features..."):
                    customer_features = build_customer_features(df)
                
                st.subheader("Customer Features Preview")
                st.dataframe(customer_features.head())
                
                # Clustering options
                st.subheader("🔧 Clustering Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    auto_clusters = st.checkbox("Auto-detect optimal clusters", value=True)
                
                with col2:
                    if not auto_clusters:
                        n_clusters = st.slider("Number of clusters", 2, 10, 5)
                
                if st.button("🚀 Train Clustering Model"):
                    if auto_clusters:
                        # Find optimal clusters
                        feature_cols = [
                            'total_spent', 'avg_order_value', 'consistency_score',
                            'recency_days', 'total_transactions', 'product_diversity',
                            'purchase_rhythm_score'
                        ]
                        clustering_data = customer_features[feature_cols].fillna(customer_features[feature_cols].median())
                        scaler = RobustScaler()
                        scaled_features = scaler.fit_transform(clustering_data)
                        
                        optimal_k, scores = find_optimal_clusters(scaled_features)
                        st.info(f"🎯 Optimal number of clusters detected: {optimal_k}")
                        n_clusters = optimal_k
                    
                    # Train model
                    final_features, model, scaler = train_clustering_model(customer_features, n_clusters)
                    
                    # Display results
                    st.subheader("📈 Training Results")
                    cluster_counts = final_features['cluster'].value_counts().sort_index()
                    
                    fig = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        title="Customer Distribution by Cluster",
                        labels={'x': 'Cluster', 'y': 'Number of Customers'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Prediction":
        st.header("🔮 Customer Segment Prediction")
        
        # Load model
        model, scaler, features = load_model_artifacts()
        
        if model is None:
            st.warning("⚠️ No trained model found. Please train a model first in the 'Data Upload & Training' tab.")
            return
        
        st.success("✅ Model loaded successfully!")
        
        # Input form
        st.subheader("Enter Customer Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_spent = st.number_input("Total Spent ($)", min_value=0.0, value=500.0)
            avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=50.0)
            consistency_score = st.slider("Consistency Score", 0.0, 1.0, 0.5)
            recency_days = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
        
        with col2:
            total_transactions = st.number_input("Total Transactions", min_value=0, value=10)
            product_diversity = st.number_input("Product Diversity", min_value=0, value=5)
            purchase_rhythm_score = st.slider("Purchase Rhythm Score", 0.0, 1.0, 0.5)
        
        if st.button("🎯 Predict Customer Segment"):
            input_features = [
                total_spent, avg_order_value, consistency_score,
                recency_days, total_transactions, product_diversity,
                purchase_rhythm_score
            ]
            
            cluster = predict_customer_cluster(model, scaler, input_features)
            
            if cluster is not None:
                st.success(f"🎯 **Predicted Cluster: {cluster}**")
                
                # Show cluster characteristics
                if features is not None:
                    cluster_data = features[features['cluster'] == cluster]
                    if len(cluster_data) > 0:
                        st.subheader(f"Cluster {cluster} Characteristics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Spending", f"${cluster_data['total_spent'].mean():.2f}")
                        with col2:
                            st.metric("Avg Recency", f"{cluster_data['recency_days'].mean():.0f} days")
                        with col3:
                            st.metric("Avg Transactions", f"{cluster_data['total_transactions'].mean():.1f}")
                        with col4:
                            dominant_stage = cluster_data['lifecycle_stage'].mode()[0]
                            st.metric("Dominant Lifecycle", dominant_stage)
    
    elif page == "Business Intelligence":
        st.header("📊 Business Intelligence Dashboard")
        
        # Load features
        _, _, features = load_model_artifacts()
        
        if features is None:
            st.warning("⚠️ No trained model found. Please train a model first.")
            return
        
        # Generate cluster profiles
        profiles = generate_cluster_profiles(features)
        
        st.subheader("🎯 Customer Segment Profiles")
        
        # Display profiles
        for cluster_name, profile in profiles.items():
            with st.expander(f"{cluster_name}: {profile['segment_name']} ({profile['size']} customers)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Spending", f"${profile['avg_spent']:.2f}")
                    st.metric("Average Recency", f"{profile['avg_recency']:.0f} days")
                
                with col2:
                    st.metric("Average Transactions", f"{profile['avg_transactions']:.1f}")
                    st.metric("Average Order Value", f"${profile['avg_order_value']:.2f}")
                
                with col3:
                    st.metric("Dominant Lifecycle", profile['dominant_lifecycle'])
                    st.metric("Consistency Score", f"{profile['consistency_score']:.2f}")
        
        # Visualizations
        st.subheader("📈 Cluster Analysis")
        
        # Cluster size distribution
        cluster_sizes = features['cluster'].value_counts().sort_index()
        fig1 = px.pie(
            values=cluster_sizes.values,
            names=[f'Cluster {i}' for i in cluster_sizes.index],
            title="Customer Distribution by Cluster"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Spending vs Recency scatter plot
        fig2 = px.scatter(
            features,
            x='recency_days',
            y='total_spent',
            color='cluster',
            title="Customer Clusters: Spending vs Recency",
            labels={'recency_days': 'Recency (days)', 'total_spent': 'Total Spent ($)'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Lifecycle stage distribution
        lifecycle_dist = features.groupby(['cluster', 'lifecycle_stage']).size().unstack(fill_value=0)
        fig3 = px.bar(
            lifecycle_dist,
            title="Lifecycle Stage Distribution by Cluster",
            labels={'value': 'Number of Customers', 'index': 'Cluster'}
        )
        st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
