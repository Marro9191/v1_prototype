import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import re
import io

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

# Default CSV data as a string
default_csv_data = """﻿date,image,SKU,promo,category,product,performance,returns,ratings,reviews,1st Page Rank,Sales
20/01/2025,https://www.amazon.co.uk/Oral-B-Electric-Toothbrush-Travel-Designed/dp/B0DNG35BVM,1,12345,tootbrush,Jenny’s Electronic Toothbrush ,150,5,5,3000,100,1
21/01/2025,,2,123123,hygiene,Competitor Toothbrush  ,120,3,5,200,5,2
22/01/2025,,3,2334234,hygiene,Jenny’s Electronic Toothbrush,145,2,5,400,10,3
23/01/2025,,4,656,hygiene,Jenny’s Electronic Toothbrush,145,2,5,30,30,4
24/01/2025,,5,345345,hygiene,Jenny’s Electronic Toothbrush,145,2,5,10,12,5
25/01/2025,,6,34535,hygiene,Jenny’s Electronic Toothbrush,145,2,5,11,39,6
26/01/2025,,7,34555,hygiene,Jenny’s Electronic Toothbrush,145,2,5,5,100,7
27/01/2025,,8,2342,hygiene,Jenny’s Electronic Toothbrush,145,2,5,5,121,8
28/01/2025,,9,2345,hygiene,Jenny’s Electronic Toothbrush,145,2,5,5,2,9
29/01/2025,,10,23422,hygiene,Jenny’s Electronic Toothbrush,145,2,5,5,34,10
30/01/2025,,11,23422,hygiene,Jenny’s Electronic Toothbrush,145,2,3,5,39,11
31/01/2025,,12,234324,hygiene,Jenny’s Electronic Toothbrush,145,2,3,5,55,12
31/01/2025,,13,2423,hygiene,Jenny’s Electronic Toothbrush,145,2,3,5,56,13
20/02/2025,,14,443,hygiene,Jenny’s Electronic Toothbrush,145,2,3,5,45,14
21/02/2025,,15,35656,hygiene,Jenny’s Electronic Toothbrush,145,2,3,5,343,15
22/02/2025,,16,56563,hygiene,Jenny’s Electronic Toothbrush,145,2,3,5,32,16
23/02/2025,,17,345345,hygiene,Jenny’s Electronic Toothbrush,145,2,3,5,234,17
24/02/2025,,18,6553,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,120,2234,18
25/02/2025,,19,453,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,120,45,19
26/02/2025,,20,34576,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,23,20
27/02/2025,,21,4545,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,34,21
28/02/2025,,22,4566,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,65,22
01/03/2025,,23,353456,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,543,23
02/03/2025,,24,656756,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,13,24
03/03/2025,,25,754646,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,34,25
04/03/2025,,26,345432,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,33,26
05/03/2025,,27,34535,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,66,27
06/03/2025,,28,4564,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,77,28
07/03/2025,,29,4567,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,90,29
08/03/2025,,30,45646,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,31,30
09/03/2025,,31,445667,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,34,31
10/03/2025,,32,2234,tootbrush,Jenny’s Electronic Toothbrush,145,2,3,60,31,32"""

# Insight Conversation
if menu == "Insight Conversation":
    st.title("📄 Comcore Prototype v1")
    st.write(
        "Ask analytical questions about the data. Supported formats: .csv, "
        "and you can also visualize the data with customizable charts. "
        "Default data is pre-loaded."
    )

    # Load default CSV data if no file is uploaded
    if 'df' not in st.session_state:
        df = pd.read_csv(io.StringIO(default_csv_data))
        st.session_state.df = df
    else:
        df = st.session_state.df

    # Default query and trigger
    if 'question' not in st.session_state:
        st.session_state.question = "What were the total number of reviews per month for all categories?"
    question = st.text_area(
        "Now ask a question about the document!",
        value=st.session_state.question,
        placeholder="Example: What were total number of reviews per month for toothbrush category? Or Which SKU had most Sales?",
    )

    # Update session state when question changes
    if question != st.session_state.question:
        st.session_state.question = question

    if df is not None and question:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')

        # Check for parsing issues and data integrity
        if df['date'].isna().all():
            st.warning("No valid dates found in the 'date' column. Please ensure dates are in DD/MM/YYYY format.")
            st.stop()
        st.write(f"Loaded {len(df)} rows from CSV.")  # Debug row count

        # Create month_year column before filtering
        df['month_year'] = df['date'].dt.strftime('%B %Y')

        # Normalize category names
        df['category'] = df['category'].str.lower().replace("tootbrush", "toothbrush")

        # Determine category filter based on query
        category_filter = None
        if "toothbrush" in question.lower():
            category_filter = "toothbrush"
        elif "all categories" in question.lower() or "all" in question.lower():
            category_filter = None  # Default to all categories
        else:
            category_filter = None  # Default to all categories if not specified

        # Apply category filter
        df_filtered = df if category_filter is None else df[df['category'] == category_filter]

        # Single OpenAI response with forced use of grouped data
        if "total number of reviews per month" in question.lower():
            # Group by month_year and category, sum all reviews
            monthly_reviews = df_filtered.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()
            # Debug the grouped data
            st.write("Grouped Data for Reviews (monthly_reviews):")
            st.write(monthly_reviews)
            openai_data = monthly_reviews.to_string()
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Based on the provided data, provide a concise summary of the total number of reviews per month for all categories. "
                        f"Use the following grouped data with columns: {list(monthly_reviews.columns)}. "
                        f"Data:\n{openai_data}\n\n---\n\n {question} Highlight the peak month and overall trend in 1-2 sentences."
                    )
                }
            ]
        else:
            # Handle "most" or "least" queries
            metric = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ["sales", "sale"]):
                    metric = col
                    break
            if not metric:
                metric = "reviews"  # Default to reviews if no sales column found
                st.warning(f"Metric '{metric}' used as default since 'sales' not found in the dataset.")
            grouped_data = df_filtered.groupby(['month_year', 'SKU'])[metric].sum().reset_index()
            openai_data = grouped_data.to_string()
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Here's the grouped data with columns: {list(grouped_data.columns)}. "
                        f"Data:\n{openai_data}\n\n---\n\n {question} Provide a concise response (e.g., 'The SKU with most {metric} is SKU 1 with 100.'). Use only this data."
                    )
                }
            ]
        stream = client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
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
            # Apply category filter
            df_filtered = df if category_filter is None else df[df['category'] == category_filter]

            # Group by month_year and category, sum all reviews
            monthly_reviews = df_filtered.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()

            # Remove duplicates in Analysis Results by ensuring unique combinations
            seen = set()
            unique_results = []
            for index, row in monthly_reviews.iterrows():
                key = (row['month_year'], row['category'])
                if key not in seen:
                    unique_results.append(row)
                    seen.add(key)
            monthly_reviews = pd.DataFrame(unique_results)

            st.subheader("Analysis Results")
            # Display as a table for better readability
            st.table(monthly_reviews.style.format({'reviews': '{:,.0f}'}).set_properties(**{'text-align': 'center'}))

            # Generate automatic bar chart with different colors for each category
            colors = {'toothbrush': '#FF6B6B', 'hygiene': '#4ECDC4'}  # Define colors for categories
            data_traces = []
            unique_months = sorted(monthly_reviews['month_year'].unique())  # Sort months chronologically

            for cat in monthly_reviews['category'].unique():
                cat_data = monthly_reviews[monthly_reviews['category'] == cat]
                data_traces.append(go.Bar(
                    x=unique_months,
                    y=[cat_data[cat_data['month_year'] == month]['reviews'].sum() if month in cat_data['month_year'].values else 0 for month in unique_months],
                    name=cat.capitalize(),
                    marker_color=colors.get(cat, '#45B7D1')  # Default color if category not in colors dict
                ))

            chart_title = f"Total Reviews Per Month by {'Toothbrush' if category_filter == 'toothbrush' else 'Category'}"
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
        elif any(word in question.lower() for word in ["most", "least"]):
            # Dynamically infer entity and metric
            entity = "SKU" if "sku" in question.lower() else "product"
            metric = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ["sales", "sale"]):
                    metric = col
                    break
            if not metric:
                metric = "reviews"  # Default to reviews if no sales column found
                st.warning(f"Metric '{metric}' used as default since 'sales' not found in the dataset.")
            group_column = entity.lower() if entity.lower() in df.columns else "SKU"  # Default to SKU if entity not found
            if group_column not in df.columns:
                st.warning(f"Grouping column '{group_column}' not found in the dataset.")
                st.stop()

            # Extract month and year for per-month analysis
            df['month_year'] = df['date'].dt.strftime('%B %Y')

            # Group by month_year and the entity, sum the metric
            entity_metrics = df.groupby(['month_year', group_column])[metric].sum().reset_index()

            # Check if data is valid
            if entity_metrics.empty or entity_metrics[metric].isna().all():
                st.warning(f"No valid {metric} data available for {entity}s.")
                st.stop()

            # Analyze per month
            st.subheader("Analysis Results")
            for month_year in entity_metrics['month_year'].unique():
                month_data = entity_metrics[entity_metrics['month_year'] == month_year]

                # Find entity with most metric for this month
                max_value = month_data[metric].max()
                most_entities = month_data[month_data[metric] == max_value][group_column].tolist()
                most_entities_str = ", ".join(most_entities) if len(most_entities) > 1 else most_entities[0]

                # Find entity with least metric for this month (excluding 0)
                min_value = month_data[month_data[metric] > 0][metric].min() if (month_data[metric] > 0).any() else 0
                least_entities = month_data[month_data[metric] == min_value][group_column].tolist() if min_value > 0 else [None]
                least_entities_str = ", ".join(filter(None, least_entities)) if len(least_entities) > 1 else (least_entities[0] if least_entities[0] else "None")

                st.write(f"{month_year}: Most {metric}: {most_entities_str} ({max_value}), Least {metric}: {least_entities_str} ({min_value if min_value > 0 else 0})")

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
            stream = client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
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
