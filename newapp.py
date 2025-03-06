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
                    "category": product["productType"] if product["productType"] else "Uncategorized"
                })
        return pd.DataFrame(product_data)
    except Exception as e:
        st.error(f"Error fetching Shopify data: {str(e)}")
        return pd.DataFrame()

# Callback function to handle file upload
def handle_upload():
    uploaded_file = st.session_state.get("insight_uploader")
    if uploaded_file:
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = set()
        if uploaded_file.name not in st.session_state.uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_insight = df
                st.session_state.uploaded_files.add(uploaded_file.name)
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.messages_insight.append({"role": "user", "content": f"Uploaded CSV file: {uploaded_file.name}"})
                st.session_state.messages_insight.append({"role": "assistant", "content": "Great! I’ve loaded your CSV file. Feel free to ask questions about it!"})
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")

# Default CSV data (removed BOM from 'date' column)
default_csv_data = """date,image,SKU,promo,category,product,performance,returns,ratings,reviews,1st Page Rank,Sales
20/01/2025,https://www.amazon.co.uk/Oral-B-Electric-Toothbrush-Travel-Designed/dp/B0DNG35BVM,1,12345,toothbrush,Jenny’s Electronic Toothbrush,150,5,5,3000,100,1
21/01/2025,,2,123123,hygiene,Competitor Toothbrush,120,3,5,200,5,2
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
24/02/2025,,18,6553,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,120,2234,18
25/02/2025,,19,453,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,120,45,19
26/02/2025,,20,34576,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,23,20
27/02/2025,,21,4545,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,34,21
28/02/2025,,22,4566,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,65,22
01/03/2025,,23,353456,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,543,23
02/03/2025,,24,656756,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,13,24
03/03/2025,,25,754646,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,34,25
04/03/2025,,26,345432,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,33,26
05/03/2025,,27,34535,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,66,27
06/03/2025,,28,4564,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,77,28
07/03/2025,,29,4567,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,90,29
08/03/2025,,30,45646,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,31,30
09/03/2025,,31,445667,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,34,31
10/03/2025,,32,2234,toothbrush,Jenny’s Electronic Toothbrush,145,2,3,60,31,32"""

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Insight Conversation", "Shopify Catalog Analysis"])

# Initialize session state
if "messages_insight" not in st.session_state:
    st.session_state.messages_insight = []
if "messages_shopify" not in st.session_state:
    st.session_state.messages_shopify = []
if "df_insight" not in st.session_state:
    st.session_state.df_insight = pd.read_csv(io.StringIO(default_csv_data))
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Custom CSS
st.markdown(
    """
    <style>
    .main-content {
        padding-bottom: 120px;
        z-index: 1;
        min-height: 100vh;
    }
    .input-wrapper {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px;
        z-index: 1002;
        border-top: 1px solid #ccc;
    }
    .stFileUploader {
        margin-bottom: 5px;
    }
    .stChatInput {
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Insight Conversation Tab
if menu == "Insight Conversation":
    st.title("📄 Comcore Prototype v1")
    st.write("Chat with me about your data! Upload a CSV or ask about reviews, sales, or specific months.")

    # Display chat messages
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        for message in st.session_state.messages_insight:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Input section
    with st.container():
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        st.file_uploader("Upload a CSV file", type=["csv"], key="insight_uploader", on_change=handle_upload)
        if prompt := st.chat_input("Ask me about your data!"):
            st.session_state.messages_insight.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Process data
            df = st.session_state.df_insight.copy()
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
            if df['date'].isna().all():
                st.error("Invalid date format in 'date' column. Please use DD/MM/YYYY.")
                st.stop()
            df['month_year'] = df['date'].dt.strftime('%B %Y')
            df['category'] = df['category'].fillna('Uncategorized').str.lower().replace("tootbrush", "toothbrush")

            # Category filter
            category_filter = "toothbrush" if "toothbrush" in prompt.lower() else None
            df_filtered = df if category_filter is None else df[df['category'] == category_filter]

            # Query processing
            if "total number of reviews per month" in prompt.lower():
                if 'reviews' not in df.columns:
                    st.error("No 'reviews' column in the data.")
                    st.stop()
                monthly_reviews = df_filtered.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"Summarize total reviews per month:\n{monthly_reviews.to_string()}"}]
                )
                st.session_state.messages_insight.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)
                    st.table(monthly_reviews)

                    fig = go.Figure([go.Bar(
                        x=monthly_reviews['month_year'],
                        y=monthly_reviews['reviews'],
                        marker_color='#FF6B6B'
                    )])
                    fig.update_layout(title="Total Reviews Per Month", xaxis_title="Month", yaxis_title="Reviews")
                    st.plotly_chart(fig)

            elif "compared to" in prompt.lower() and "reviews" in prompt.lower():
                if 'reviews' not in df.columns:
                    st.error("No 'reviews' column in the data.")
                    st.stop()
                months = re.findall(r'(January|February|March|April|May|June|July|August|September|October|November|December)', prompt, re.IGNORECASE)
                if len(months) >= 2:
                    month1, month2 = months[:2]
                    month1_data = df_filtered[df_filtered['month_year'].str.contains(month1, case=False, na=False)]
                    month2_data = df_filtered[df_filtered['month_year'].str.contains(month2, case=False, na=False)]
                    month1_reviews = month1_data['reviews'].sum()
                    month2_reviews = month2_data['reviews'].sum()

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"Compare reviews: {month1} ({month1_reviews}) vs {month2} ({month2_reviews})"}]
                    )
                    st.session_state.messages_insight.append({"role": "assistant", "content": response.choices[0].message.content})
                    with st.chat_message("assistant"):
                        st.write(response.choices[0].message.content)
                        fig = go.Figure([go.Bar(x=[month1, month2], y=[month1_reviews, month2_reviews])])
                        fig.update_layout(title=f"Reviews: {month1} vs {month2}", yaxis_title="Reviews")
                        st.plotly_chart(fig)

            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "I don’t understand your question. Ask about reviews, sales, or months!"}]
                )
                st.session_state.messages_insight.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)

# Shopify Catalog Analysis Tab
elif menu == "Shopify Catalog Analysis":
    st.title("🛒 Shopify Catalog Analysis")
    st.write("Ask about your Shopify catalog!")

    for message in st.session_state.messages_shopify:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask about your Shopify catalog!"):
        st.session_state.messages_shopify.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Fetching Shopify data..."):
            df = fetch_shopify_products()

        if df.empty:
            st.session_state.messages_shopify.append({"role": "assistant", "content": "Couldn’t fetch Shopify data. Check your API credentials."})
            with st.chat_message("assistant"):
                st.write("Couldn’t fetch Shopify data. Check your API credentials.")
        else:
            if "out of stock" in prompt.lower():
                out_of_stock = df[df['inventory_quantity'] == 0]
                count = len(out_of_stock)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"{count} products out of stock:\n{out_of_stock[['title', 'sku']].to_string()}"}]
                )
                st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)
                    fig = go.Figure([go.Pie(labels=['In Stock', 'Out of Stock'], values=[len(df) - count, count])])
                    fig.update_layout(title="Stock Status")
                    st.plotly_chart(fig)
            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Ask about stock levels or updates!"}]
                )
                st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)
