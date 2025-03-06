import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import re

# Add sidebar with menu items
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Insight Conversation", "Shopify Catalog Analysis"])

# Initialize OpenAI client
try:
    openai_api_key = st.secrets["openai"]["api_key"]
    client = OpenAI(api_key=openai_api_key)
except KeyError:
    st.error("Please add your OpenAI API key to `.streamlit/secrets.toml` under the key `openai.api_key`.")
    st.stop()

# Function to fetch Shopify products using GraphQL
def fetch_shopify_products():
    try:
        shopify_domain = st.secrets["shopify"]["domain"]
        access_token = st.secrets["shopify"]["access_token"]
        api_version = "2024-10"

        url = f"https://{shopify_domain}/admin/api/{api_version}/graphql.json"
        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": access_token
        }
        
        query = """
        query {
          products(first: 100) {
            edges {
              node {
                id
                title
                productType
                variants(first: 10) {
                  edges {
                    node {
                      id
                      sku
                      price
                      inventoryQuantity
                    }
                  }
                }
                createdAt
                updatedAt
              }
            }
          }
        }
        """
        
        response = requests.post(url, headers=headers, json={"query": query})
        response.raise_for_status()
        
        data = response.json()["data"]["products"]["edges"]
        
        product_data = []
        for edge in data:
            product = edge["node"]
            for variant_edge in product["variants"]["edges"]:
                variant = variant_edge["node"]
                product_data.append({
                    "product_id": product["id"],
                    "title": product["title"],
                    "variant_id": variant["id"],
                    "sku": variant["sku"],
                    "price": float(variant["price"]),
                    "inventory_quantity": variant["inventoryQuantity"],
                    "created_at": pd.to_datetime(product["createdAt"]),
                    "updated_at": pd.to_datetime(product["updatedAt"]),
                    "category": product["productType"] or "Uncategorized"
                })
        return pd.DataFrame(product_data)
    except Exception as e:
        st.error(f"Error fetching Shopify data: {str(e)}")
        return pd.DataFrame()

# Insight Conversation
if menu == "Insight Conversation":
    st.title("📄 Comcore Prototype v1")
    st.write(
        "Upload CSV file below and ask analytical questions. "
        "Supported formats: .csv, "
        "and you can also visualize the data with customizable charts. "
        "Please note it has to be UTF-8 encoded."
    )

    uploaded_file = st.file_uploader("Upload a document (.csv)", type="csv")
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Example: What were total number of reviews per month for toothbrush category? Or Provide me item with least reviews and most reviews",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        df = pd.read_csv(uploaded_file)
        document = df.to_string()
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')

        # Check for parsing issues
        if df['date'].isna().all():
            st.warning("No valid dates found in the 'date' column. Please ensure dates are in DD/MM/YYYY format.")
            st.stop()

        # Create month_year column before filtering
        df['month_year'] = df['date'].dt.strftime('%B %Y')

        # Normalize category names
        df['category'] = df['category'].str.lower().replace("tootbrush", "toothbrush")

        # Single OpenAI response
        messages = [{"role": "user", "content": f"Here's a document: {document} \n\n---\n\n {question} Provide a concise response listing the totals per month and category (e.g., 'January 2025: Toothbrush 3000, Hygiene 746.')."}]
        stream = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True)
        st.subheader("Response")
        st.write_stream(stream)

        # Custom analysis for review comparison (last month vs this month)
        if "reviews" in question.lower() and "last month" in question.lower() and "this month" in question.lower():
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            last_month_year = current_year - 1 if current_month == 1 else current_year
            last_month = 12 if current_month == 1 else current_month - 1

            category = "toothbrush" if "toothbrush" in question.lower() else None
            df_filtered = df[df['category'].str.lower().str.contains("toot?brush", na=False)] if category else df

            this_month_data = df_filtered[
                (df_filtered['date'].dt.month == current_month) & 
                (df_filtered['date'].dt.year == current_year)
            ]
            last_month_data = df_filtered[
                (df_filtered['date'].dt.month == last_month) & 
                (df_filtered['date'].dt.year == last_month_year)
            ]

            this_month_reviews = this_month_data['reviews'].sum() if 'reviews' in this_month_data.columns else 0
            last_month_reviews = last_month_data['reviews'].sum() if 'reviews' in last_month_data.columns else 0

            st.subheader("Analysis Results")
            st.write(f"This Month: {this_month_reviews} reviews")
            st.write(f"Last Month: {last_month_reviews} reviews")

            fig = go.Figure(data=[
                go.Bar(x=['Last Month', 'This Month'], y=[last_month_reviews, this_month_reviews], marker_color=['#FF6B6B', '#4ECDC4'])
            ])
            fig.update_layout(
                title=f"Reviews Comparison - {category if category else 'All Categories'}",
                xaxis_title="Period",
                yaxis_title="Number of Reviews",
                height=500,
                width=700
            )
            st.plotly_chart(fig)

        # Custom analysis for total number of reviews per month
        elif "total number of reviews per month" in question.lower():
            category = "toothbrush" if "toothbrush" in question.lower() else None
            df_filtered = df[df['category'].str.lower().str.contains("toot?brush", na=False)] if category else df

            # Group by month_year and category, sum all reviews
            monthly_reviews = df_filtered.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()

            # Remove duplicates in Analysis Results by using a set of unique combinations
            seen = set()
            unique_results = []
            for index, row in monthly_reviews.iterrows():
                key = (row['month_year'], row['category'])
                if key not in seen:
                    unique_results.append(row)
                    seen.add(key)
            monthly_reviews = pd.DataFrame(unique_results)

            st.subheader("Analysis Results")
            for index, row in monthly_reviews.iterrows():
                st.write(f"{row['month_year']} - {row['category'].capitalize()}: {row['reviews']} reviews")

            # Generate automatic bar chart with different colors for each category
            colors = {'toothbrush': '#FF6B6B', 'hygiene': '#4ECDC4'}  # Define colors for categories
            data_traces = []
            unique_months = monthly_reviews['month_year'].unique()

            for cat in monthly_reviews['category'].unique():
                cat_data = monthly_reviews[monthly_reviews['category'] == cat]
                data_traces.append(go.Bar(
                    x=unique_months,
                    y=[cat_data[cat_data['month_year'] == month]['reviews'].sum() if month in cat_data['month_year'].values else 0 for month in unique_months],
                    name=cat.capitalize(),
                    marker_color=colors.get(cat, '#45B7D1')  # Default color if category not in colors dict
                ))

            chart_title = "Total Reviews Per Month by Category"
            fig = go.Figure(data=data_traces)
            fig.update_layout(
                title=chart_title,
                xaxis_title="Month",
                yaxis_title="Number of Reviews",
                height=500,
                width=700,
                barmode='group',  # Group bars by category
                showlegend=True
            )
            st.plotly_chart(fig)

        # Custom analysis for most and least values dynamically
        elif any(word in question.lower() for word in ["most", "least"]) and any(metric in question.lower() for metric in ["reviews", "sales"]):
            # Dynamically infer entity and metric
            entity = "SKU" if "item" in question.lower() or "sku" in question.lower() else "product"
            metric = "reviews" if "reviews" in question.lower() else "sales" if "sales" in question.lower() else None
            if not metric or metric not in df.columns:
                st.warning(f"Metric '{metric}' not found in the dataset.")
                st.stop()

            group_column = entity.lower() if entity.lower() in df.columns else "SKU"  # Default to SKU if entity not found
            if group_column not in df.columns:
                st.warning(f"Grouping column '{group_column}' not found in the dataset.")
                st.stop()

            # Extract month and year for per-month analysis
            df['month_year'] = df['date'].dt.strftime('%B %Y')

            # Group by month_year and the entity, sum the metric
            entity_metrics = df.groupby(['month_year', group_column])['reviews'].sum().reset_index()

            # Check if data is valid
            if entity_metrics.empty or entity_metrics['reviews'].isna().all():
                st.warning(f"No valid {metric} data available for {entity}s.")
                st.stop()

            # Analyze per month
            st.subheader("Analysis Results")
            for month_year in entity_metrics['month_year'].unique():
                month_data = entity_metrics[entity_metrics['month_year'] == month_year]

                # Find entity with most metric for this month
                max_value = month_data['reviews'].max()
                most_entities = month_data[month_data['reviews'] == max_value][group_column].head(1).iloc[0]

                # Find entity with least metric for this month (excluding 0)
                min_value = month_data[month_data['reviews'] > 0]['reviews'].min() if (month_data['reviews'] > 0).any() else 0
                least_entities = month_data[month_data['reviews'] == min_value][group_column].head(1).iloc[0] if min_value > 0 else None

                st.write(f"{month_year}: Most {metric}: {most_entities} ({max_value}), Least {metric}: {least_entities if least_entities else 'None'} ({min_value if min_value > 0 else 0})")

        # General visualization options
        st.subheader("Custom Visualization")
        if not df.empty:
            chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie", "Scatter", "Area"])
            x_col = st.selectbox("X-axis", df.columns)
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_cols) > 0:
                y_col = st.selectbox("Y-axis", numeric_cols)
                color_option = st.selectbox("Color by", ["Single Color"] + df.columns.tolist())
                color = st.color_picker("Pick a color", "#00f900") if color_option == "Single Color" else color_option
                chart_title = st.text_input("Chart Title", "Data Visualization")

                if st.button("Generate Chart"):
                    fig = go.Figure()
                    if chart_type == "Bar":
                        fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], marker_color=color if color_option == "Single Color" else None))
                    elif chart_type == "Line":
                        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines', line=dict(color=color if color_option == "Single Color" else None)))
                    elif chart_type == "Pie":
                        pie_data = df.groupby(x_col)[y_col].sum()
                        fig.add_trace(go.Pie(labels=pie_data.index, values=pie_data.values))
                    elif chart_type == "Scatter":
                        fig.add_trace(go.Scatter(
                            x=df[x_col], y=df[y_col], mode='markers',
                            marker=dict(color=df[color] if color_option != "Single Color" else color, size=10)
                        ))
                    elif chart_type == "Area":
                        fig.add_trace(go.Scatter(
                            x=df[x_col], y=df[y_col], fill='tozeroy',
                            line=dict(color=color if color_option == "Single Color" else None)
                        ))

                    fig.update_layout(title=chart_title, xaxis_title=x_col, yaxis_title=y_col, height=500, width=700)
                    st.plotly_chart(fig)
            else:
                st.warning("No numeric columns available for charting.")
        else:
            st.warning("The uploaded data is empty.")

# Shopify Catalog Analysis
elif menu == "Shopify Catalog Analysis":
    st.title("🛒 Shopify Catalog Analysis")
    st.write(
        "Ask analytical questions about your Shopify product catalog. "
        "Data is fetched directly from your Shopify store when you submit a question."
    )

    question = st.text_area(
        "Ask a question about your Shopify catalog!",
        placeholder="Example: What were total number of products updated last month compared to this month for Electronics category?",
    )

    if question:
        with st.spinner("Fetching Shopify catalog data via GraphQL..."):
            df = fetch_shopify_products()

        if df.empty:
            st.warning("No data fetched from Shopify. Check your API credentials.")
        else:
            document = df.to_string()
            messages = [{"role": "user", "content": f"Here's the Shopify catalog data: {document} \n\n---\n\n {question}"}]
            stream = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True)
            st.subheader("Response")
            st.write_stream(stream)

            # Custom analysis for product updates comparison
            if "last month" in question.lower() and "this month" in question.lower():
                current_date = datetime.now()
                current_month = current_date.month
                current_year = current_date.year
                last_month_year = current_year - 1 if current_month == 1 else current_year
                last_month = 12 if current_month == 1 else current_month - 1

                category = "Electronics" if "electronics" in question.lower() else None
                df_filtered = df[df['category'].str.lower() == category.lower()] if category else df

                this_month_data = df_filtered[
                    (df_filtered['updated_at'].dt.month == current_month) & 
                    (df_filtered['updated_at'].dt.year == current_year)
                ]
                last_month_data = df_filtered[
                    (df_filtered['updated_at'].dt.month == last_month) & 
                    (df_filtered['updated_at'].dt.year == last_month_year)
                ]

                this_month_count = this_month_data.shape[0]
                last_month_count = last_month_data.shape[0]

                st.subheader("Analysis Results")
                st.write(f"This Month: {this_month_count} products")
                st.write(f"Last Month: {last_month_count} products")

                fig = go.Figure(data=[
                    go.Bar(x=['Last Month', 'This Month'], y=[last_month_count, this_month_count], marker_color=['#FF6B6B', '#4ECDC4'])
                ])
                fig.update_layout(
                    title=f"Product Updates Comparison - {category if category else 'All Categories'}",
                    xaxis_title="Period",
                    yaxis_title="Number of Products Updated",
                    height=500,
                    width=700
                )
                st.plotly_chart(fig)

            # General visualization options
            st.subheader("Custom Visualization")
            if not df.empty:
                chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie", "Scatter", "Area"])
                x_col = st.selectbox("X-axis", df.columns)
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numeric_cols) > 0:
                    y_col = st.selectbox("Y-axis", numeric_cols)
                    color_option = st.selectbox("Color by", ["Single Color"] + df.columns.tolist())
                    color = st.color_picker("Pick a color", "#00f900") if color_option == "Single Color" else color_option
                    chart_title = st.text_input("Chart Title", "Shopify Data Visualization")

                    if st.button("Generate Chart"):
                        fig = go.Figure()
                        if chart_type == "Bar":
                            fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], marker_color=color if color_option == "Single Color" else None))
                        elif chart_type == "Line":
                            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines', line=dict(color=color if color_option == "Single Color" else None)))
                        elif chart_type == "Pie":
                            pie_data = df.groupby(x_col)[y_col].sum()
                            fig.add_trace(go.Pie(labels=pie_data.index, values=pie_data.values))
                        elif chart_type == "Scatter":
                            fig.add_trace(go.Scatter(
                                x=df[x_col], y=df[y_col], mode='markers',
                                marker=dict(color=df[color] if color_option != "Single Color" else color, size=10)
                            ))
                        elif chart_type == "Area":
                            fig.add_trace(go.Scatter(
                                x=df[x_col], y=df[y_col], fill='tozeroy',
                                line=dict(color=color if color_option == "Single Color" else None)
                            ))

                        fig.update_layout(title=chart_title, xaxis_title=x_col, yaxis_title=y_col, height=500, width=700)
                        st.plotly_chart(fig)
                else:
                    st.warning("No numeric columns available for charting.")
            else:
                st.warning("The fetched Shopify data is empty.")
