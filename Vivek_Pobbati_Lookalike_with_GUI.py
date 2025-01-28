import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Streamlit interface
st.title('Customer Lookalike Recommendation System')

# File uploaders for customers, products, and transactions
customers_file = st.file_uploader("Upload Customers CSV", type=["csv"])
products_file = st.file_uploader("Upload Products CSV", type=["csv"])
transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

if customers_file and products_file and transactions_file:
    # Load the data
    customers = pd.read_csv(customers_file)
    products = pd.read_csv(products_file)
    transactions = pd.read_csv(transactions_file)

    # Merge transactions with customers and products
    data = pd.merge(transactions, customers, on='CustomerID')
    data = pd.merge(data, products, on='ProductID')

    # Feature Engineering
    # Create customer features based on their transaction history
    customer_features = data.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'count'],
        'Quantity': ['sum', 'mean'],
        'Price_x': ['mean', 'std'],
        'Category': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'  # Most frequent category
    }).reset_index()

    # Flatten the multi-level columns
    customer_features.columns = ['CustomerID', 'TotalValue_sum', 'TotalValue_mean', 'Transaction_count', 
                                 'Quantity_sum', 'Quantity_mean', 'Price_mean', 'Price_std', 'Favorite_Category']

    # Merge with customer information
    customer_features = pd.merge(customer_features, customers, on='CustomerID')

    # Handle missing values in numerical columns (replace NaN with 0 or mean)
    customer_features['Price_std'].fillna(0, inplace=True)  # Replace NaN in Price_std with 0
    customer_features['Price_mean'].fillna(customer_features['Price_mean'].mean(), inplace=True)  # Replace NaN in Price_mean with the mean

    # Encode categorical variables
    customer_features = pd.get_dummies(customer_features, columns=['Region', 'Favorite_Category'])

    # Normalize the features
    scaler = StandardScaler()
    customer_features_scaled = scaler.fit_transform(customer_features.drop(columns=['CustomerID', 'CustomerName', 'SignupDate']))

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(customer_features_scaled)

    # Create a DataFrame for the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])

    # Function to get top 3 similar customers
    def get_top_similar_customers(customer_id, similarity_df, top_n=3):
        similar_customers = similarity_df[customer_id].sort_values(ascending=False).iloc[1:top_n+1]
        return similar_customers.index.tolist(), similar_customers.values.tolist()

    # Generate recommendations for customers in the file
    customer_ids = customer_features['CustomerID'].tolist()
    selected_customer_id = st.selectbox("Select a Customer ID", customer_ids)

    if selected_customer_id:
        similar_customers, scores = get_top_similar_customers(selected_customer_id, similarity_df)
        st.write(f"Top 3 similar customers to Customer ID {selected_customer_id}:")
        for i in range(len(similar_customers)):
            st.write(f"Customer ID: {similar_customers[i]}, Similarity Score: {scores[i]}")

    # Generate recommendations for the first 20 customers
    lookalike_map = {}
    for customer_id in customer_features['CustomerID'][:20]:
        similar_customers, scores = get_top_similar_customers(customer_id, similarity_df)
        # Convert list of tuples to string representation
        lookalike_map[customer_id] = [f"Customer ID: {similar_customers[i]}, Similarity Score: {scores[i]}" for i in range(len(similar_customers))]

    # Convert the map to a DataFrame for display
    lookalike_df = pd.DataFrame(list(lookalike_map.items()), columns=['CustomerID', 'Lookalikes'])

    # Display the lookalike DataFrame
    st.write("Lookalike Recommendations for the First 20 Customers:")
    st.write(lookalike_df)

    # Allow user to download the lookalike recommendations
    lookalike_csv = lookalike_df.to_csv(index=False)
    st.download_button(
        label="Download Lookalike Recommendations CSV",
        data=lookalike_csv,
        file_name='Lookalike.csv',
        mime='text/csv'
    )
