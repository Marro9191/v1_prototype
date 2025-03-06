import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import re
import io

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

# Default CSV data as a string for Insight Conversation
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

# Main chatbot interface
st.title("📝 Comcore Chatbot")
st.write("Welcome! Ask me analytical questions about your Shopify catalog or CSV data. I’ll respond with insights and visuals when applicable.")

# Initialize session state for chat history and data
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'shopify_df' not in st.session_state:
    st.session_state.shopify_df = None
if 'insight_df' not in st.session_state:
    st.session_state.insight_df = pd.read_csv(io.StringIO(default_csv_data))

# Fetch Shopify data once and store in session state
if st.session_state.shopify_df is None:
    with st.spinner("Fetching Shopify catalog data via GraphQL..."):
        st.session_state.shopify_df = fetch_shopify_products()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Prepare response based on context (Shopify or Insight)
    df = st.session_state.shopify_df if "shopify" in user_input.lower() else st.session_state.insight_df
    if df is not None and not df.empty:
        document = df.to_string()

        # Default OpenAI prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides friendly and concise responses to analytical questions about product data. Include insights or visualizations when applicable."},
            {"role": "user", "content": f"Here's the data: {document} \n\n---\n\n {user_input}"}
        ]

        # Customize response based on specific queries
        if "which products are out of stock" in user_input.lower() and "how many" in user_input.lower():
            out_of_stock = df[df['inventory_quantity'] == 0]
            out_of_stock_count = len(out_of_stock)
            in_stock_count = len(df[df['inventory_quantity'] > 0])
            out_of_stock_list = out_of_stock[['title', 'sku']].drop_duplicates().to_dict('records')

            response_content = f"Hey there! We’ve checked your stock, and here are the products currently out of stock: {', '.join([f'{item['title']} (SKU: {item['sku']})' for item in out_of_stock_list]) if out_of_stock_count > 0 else 'great news—there are no products out of stock right now!'}. Time to restock if needed!"
            messages[1]["content"] = f"Here's the data: {document} \n\n---\n\n {user_input} Respond with: {response_content}"

            with st.chat_message("assistant"):
                st.write(response_content)

            if out_of_stock_count > 0:
                st.write(f"**Total out-of-stock products:** {out_of_stock_count}")
                for item in out_of_stock_list:
                    st.write(f"- {item['title']} (SKU: {item['sku']}) - 0 items in stock")
            else:
                st.write("**Great news! No products are out of stock.**")

            # Generate pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=['In Stock', 'Out of Stock'],
                    values=[in_stock_count, out_of_stock_count],
                    marker_colors=['#4ECDC4', '#FF6B6B'],
                    textinfo='label+percent',
                    hole=0.3
                )
            ])
            fig.update_layout(
                title="Stock Status: In Stock vs Out of Stock",
                height=500,
                width=700,
                showlegend=True
            )
            st.plotly_chart(fig)

        elif "last month" in user_input.lower() and "this month" in user_input.lower() and "reviews" in user_input.lower():
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            last_month_year = current_year - 1 if current_month == 1 else current_year
            last_month = 12 if current_month == 1 else current_month - 1

            category = "toothbrush" if "toothbrush" in user_input.lower() else None
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

            response_content = f"The total number of reviews for the {category or 'all'} category last month was {last_month_reviews}, compared to {this_month_reviews} this month!"
            messages[1]["content"] = f"Here's the data: {document} \n\n---\n\n {user_input} Respond with: {response_content}"

            with st.chat_message("assistant"):
                st.write(response_content)

            st.write(f"**This Month:** {this_month_reviews} reviews")
            st.write(f"**Last Month:** {last_month_reviews} reviews")

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

        elif "total number of reviews per month" in user_input.lower():
            monthly_reviews = df.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()
            openai_data = monthly_reviews.to_string()
            response_content = f"Here’s a quick look at the total number of reviews per month: {openai_data.split('\n')[1:]}"
            messages[1]["content"] = f"Here's the data: {document} \n\n---\n\n {user_input} Respond with: {response_content}"

            with st.chat_message("assistant"):
                st.write(response_content)

            monthly_reviews = df.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()
            seen = set()
            unique_results = []
            for index, row in monthly_reviews.iterrows():
                key = (row['month_year'], row['category'])
                if key not in seen:
                    unique_results.append(row)
                    seen.add(key)
            monthly_reviews = pd.DataFrame(unique_results)

            st.write("**Monthly Review Breakdown:**")
            st.table(monthly_reviews.style.format({'reviews': '{:,.0f}'}))

            colors = {'toothbrush': '#FF6B6B', 'hygiene': '#4ECDC4'}
            data_traces = []
            unique_months = sorted(monthly_reviews['month_year'].unique())

            for cat in monthly_reviews['category'].unique():
                cat_data = monthly_reviews[monthly_reviews['category'] == cat]
                data_traces.append(go.Bar(
                    x=unique_months,
                    y=[cat_data[cat_data['month_year'] == month]['reviews'].sum() if month in cat_data['month_year'].values else 0 for month in unique_months],
                    name=cat.capitalize(),
                    marker_color=colors.get(cat, '#45B7D1')
                ))

            fig = go.Figure(data=data_traces)
            fig.update_layout(
                title="Total Reviews Per Month by Category",
                xaxis_title="Month",
                yaxis_title="Number of Reviews",
                height=500,
                width=700,
                barmode='group',
                showlegend=True
            )
            st.plotly_chart(fig)

        else:
            # Default response for other queries
            stream = client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
            with st.chat_message("assistant"):
                st.write_stream(stream)

    else:
        with st.chat_message("assistant"):
            st.write("Sorry, I couldn’t fetch the data. Please check your API credentials or try again later.")

    # Add assistant message to chat history
    if 'response_content' in locals():
        st.session_state.chat_history.append({"role": "assistant", "content": response_content})
    else:
        st.session_state.chat_history.append({"role": "assistant", "content": "".join([msg for msg in st.session_state.chat_history[-1]["content"] if hasattr(msg, 'content')])})
