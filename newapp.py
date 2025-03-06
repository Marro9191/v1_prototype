import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import calendar

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

# Function to parse months from the query
def parse_months_from_query(question):
    # Map month names to numbers
    month_map = {name.lower(): idx for idx, name in enumerate(calendar.month_name) if name}
    month_abbr_map = {name.lower(): idx for idx, name in enumerate(calendar.month_abbr) if name}
    month_map.update(month_abbr_map)

    # Find months in the query
    question_lower = question.lower()
    months_found = []
    for month_name, month_num in month_map.items():
        if month_name in question_lower:
            months_found.append((month_name, month_num))

    # Sort by position in the query to maintain order (e.g., "January compared to February")
    months_found.sort(key=lambda x: question_lower.index(x[0]))

    # Extract years if provided (e.g., "January 2024")
    years = {}
    for month_name, month_num in months_found:
        month_idx = question_lower.index(month_name)
        remaining_text = question_lower[month_idx + len(month_name):]
        import re
        year_match = re.search(r'\b(\d{4})\b', remaining_text)
        if year_match:
            years[month_num] = int(year_match.group(1))
        else:
            years[month_num] = None

    return months_found, years

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
        placeholder="Example: What were total number of reviews last month compared to this month for tootbrush category?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        df = pd.read_csv(uploaded_file)
        document = df.to_string()

        # Parse dates with flexible formats (DD/MM/YY or DD/MM/YYYY)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y', errors='coerce', dayfirst=True)
        if df['date'].isna().all():
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
        if df['date'].isna().all():
            st.warning("No valid dates found in the 'date' column. Please use DD/MM/YY or DD/MM/YYYY format.")
            st.stop()

        # Debug: Show a representative sample (include both January and February for tootbrush)
        sample_data = df[df['category'].str.lower() == 'tootbrush'].head(10)
        st.write("Sample data (filtered for tootbrush):", sample_data)

        # Handle dynamic month comparison queries
        months_found, years = parse_months_from_query(question)
        category = next((cat for cat in ["tootbrush", "toothbrush"] if cat.lower() in question.lower()), None)

        # Check if the query involves comparing two months for reviews
        if "reviews" in question.lower() and "compared" in question.lower() and len(months_found) >= 2:
            # Get the two months to compare
            first_month_name, first_month = months_found[0]
            second_month_name, second_month = months_found[1]

            # Determine the years (default to the most recent year in the data if not specified)
            default_year = df['date'].dt.year.max() if not df['date'].dt.year.isna().all() else 2025
            first_year = years[first_month] if years[first_month] else default_year
            second_year = years[second_month] if years[second_month] else default_year

            # Filter for category
            if category:
                df_filtered = df[df['category'].str.lower() == category.lower()]
            else:
                df_filtered = df

            # Filter for the specified months
            first_month_data = df_filtered[
                (df_filtered['date'].dt.month == first_month) & 
                (df_filtered['date'].dt.year == first_year)
            ]
            second_month_data = df_filtered[
                (df_filtered['date'].dt.month == second_month) & 
                (df_filtered['date'].dt.year == second_year)
            ]

            first_month_reviews = first_month_data['reviews'].sum() if 'reviews' in first_month_data.columns else 0
            second_month_reviews = second_month_data['reviews'].sum() if 'reviews' in second_month_data.columns else 0

            # Prepare filtered data for OpenAI
            relevant_data = pd.concat([first_month_data, second_month_data])
            openai_data = relevant_data[['date', 'category', 'reviews']].to_string() if not relevant_data.empty else "No relevant data found."

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Here is a dataset with columns: {list(df.columns)}. "
                        f"Relevant data for the question (filtered for category '{category}' and months {first_month_name} {first_year}/{second_month_name} {second_year}):\n{openai_data}\n\n"
                        f"---\n\nPlease calculate the total number of reviews for {first_month_name} {first_year} "
                        f"and {second_month_name} {second_year} for the '{category}' category, and compare them. "
                        f"Sum the 'reviews' column for each month separately. Provide the totals and the difference between them."
                    )
                }
            ]
            stream = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True)
            st.subheader("Response")
            st.write_stream(stream)

            st.subheader("Analysis Results")
            st.write(f"Total Reviews in {first_month_name} {first_year}: {first_month_reviews}")
            st.write(f"Total Reviews in {second_month_name} {second_year}: {second_month_reviews}")

            fig = go.Figure(data=[
                go.Bar(
                    x=[f"{first_month_name} {first_year}", f"{second_month_name} {second_year}"],
                    y=[first_month_reviews, second_month_reviews],
                    marker_color=['#FF6B6B', '#4ECDC4']
                )
            ])
            fig.update_layout(
                title=f"Reviews Comparison - {category if category else 'All Categories'}",
                xaxis_title="Period",
                yaxis_title="Number of Reviews",
                height=500,
                width=700
            )
            st.plotly_chart(fig)

        # Handle "last month" and "this month" queries for backward compatibility
        elif "reviews" in question.lower() and "last month" in question.lower() and "this month" in question.lower():
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            last_month_year = current_year - 1 if current_month == 1 else current_year
            last_month = 12 if current_month == 1 else current_month - 1

            if category:
                df_filtered = df[df['category'].str.lower() == category.lower()]
            else:
                df_filtered = df

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

            relevant_data = pd.concat([last_month_data, this_month_data])
            openai_data = relevant_data[['date', 'category', 'reviews']].to_string() if not relevant_data.empty else "No relevant data found."

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Here is a dataset with columns: {list(df.columns)}. "
                        f"Relevant data for the question (filtered for category '{category}' and last/this month):\n{openai_data}\n\n"
                        f"---\n\nPlease calculate the total number of reviews last month (February 2025) "
                        f"and this month (March 2025) for the '{category}' category, and compare them. "
                        f"Sum the 'reviews' column for each month separately."
                    )
                }
            ]
            stream = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True)
            st.subheader("Response")
            st.write_stream(stream)

            st.subheader("Analysis Results")
            st.write(f"Total Reviews This Month: {this_month_reviews}")
            st.write(f"Total Reviews Last Month: {last_month_reviews}")

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

# Shopify Catalog Analysis (unchanged)
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
                st.write(f"Total Products Updated This Month: {this_month_count}")
                st.write(f"Total Products Updated Last Month: {last_month_count}")

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
