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

# Reset session state at startup to ensure no residual data
def reset_session_state():
    st.write("Debug: Resetting session state...")
    for key in list(st.session_state.keys()):
        if key in ["messages_insight", "df_insight", "upload_message_added", "last_uploaded_file", "messages_shopify"]:
            del st.session_state[key]
    st.session_state.messages_insight = []
    st.session_state.df_insight = pd.DataFrame()  # No preloaded data
    st.session_state.upload_message_added = {}
    st.session_state.last_uploaded_file = None
    st.session_state.messages_shopify = []
    # Clear any cached data (optional, for safety)
    st.cache_data.clear()

# Call reset at the start of the app
if "session_initialized" not in st.session_state:
    reset_session_state()
    st.session_state.session_initialized = True

# Callback function to handle file upload
def handle_upload():
    uploaded_file = st.session_state.uploaded_file
    if uploaded_file and uploaded_file.name not in st.session_state.upload_message_added:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_insight = df
        st.session_state.upload_message_added[uploaded_file.name] = True
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.messages_insight.append({"role": "user", "content": f"Uploaded CSV file: {uploaded_file.name}"})
        st.session_state.messages_insight.append({"role": "assistant", "content": "Great! I’ve loaded your CSV file. Feel free to ask questions about it!"})

# Custom CSS to enforce layout
st.markdown(
    """
    <style>
    .main-content {
        padding-bottom: 160px; /* Space for the fixed input container */
        z-index: 1;
        min-height: 100vh; /* Ensure content takes full height */
    }
    .input-wrapper {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background-color: white !important;
        padding: 10px !important;
        z-index: 1003 !important;
        border-top: 1px solid #ccc !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 0px !important; /* No gap between uploader and chat input */
    }
    .stFileUploader {
        margin-bottom: 0px !important; /* Remove space below uploader */
        padding-bottom: 0px !important;
    }
    .stChatInput {
        margin-top: 0px !important; /* Remove space above chat input */
        padding-top: 0px !important;
    }
    /* Ensure main content stays above input wrapper */
    .stApp {
        overflow: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add sidebar with menu items
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Insight Conversation", "Shopify Catalog Analysis"])

# Display chat interface based on selected tab
if menu == "Insight Conversation":
    st.title("📄 Comcore Prototype v1")
    st.write("Upload a CSV file and ask me about your data! (e.g., 'What were the total number of reviews per month?')")

    # Manual reset button for debugging
    if st.button("Reset Session"):
        reset_session_state()
        st.experimental_rerun()

    # Main container for chat messages and responses (above input)
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        # Display existing chat messages
        for message in st.session_state.messages_insight:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        # Check if no data is loaded and prompt user
        if st.session_state.df_insight.empty:
            st.write("Please upload a CSV file to start analyzing your data.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Input container at the bottom with CSS styling
    with st.container():
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        # File uploader placed just above the chat input
        st.file_uploader("Upload a CSV file", type=["csv"], key="uploaded_file", on_change=handle_upload, help="Upload your data file to analyze.")
        
        # Chat input for Insight Conversation, directly below the file uploader
        if prompt := st.chat_input("Ask me about your data! (e.g., 'What were the total number of reviews per month?')"):
            st.session_state.messages_insight.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Load and process data
            df = st.session_state.df_insight
            if df.empty:
                st.session_state.messages_insight.append({"role": "assistant", "content": "Please upload a CSV file before asking questions about the data."})
                with st.chat_message("assistant"):
                    st.write("Please upload a CSV file before asking questions about the data.")
                st.stop()

            # Validate data and preprocess
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
            if df['date'].isna().all():
                st.warning("No valid dates found in the 'date' column. Please ensure dates are in DD/MM/YYYY format.")
                st.stop()
            st.write(f"Loaded {len(df)} rows from uploaded CSV.")  # Debug row count
            df['month_year'] = df['date'].dt.strftime('%B %Y')
            df['category'] = df['category'].str.lower().replace("tootbrush", "toothbrush")

            # Dynamic category filtering
            category_filter = None
            categories = df['category'].unique()
            for cat in categories:
                if cat in prompt.lower():
                    category_filter = cat
                    break
            if not category_filter and ("all categories" in prompt.lower() or "all" in prompt.lower()):
                category_filter = None
            df_filtered = df if category_filter is None else df[df['category'] == category_filter]

            # Process the query
            if "compared to" in prompt.lower() and "reviews" in prompt.lower():
                months = re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', prompt, re.IGNORECASE)
                if len(months) >= 2:
                    month1, month2 = months[0], months[1]
                    month1_data = df_filtered[df_filtered['month_year'].str.contains(month1, case=False, na=False)]
                    month2_data = df_filtered[df_filtered['month_year'].str.contains(month2, case=False, na=False)]

                    month1_reviews = month1_data['reviews'].sum() if 'reviews' in month1_data.columns else 0
                    month2_reviews = month2_data['reviews'].sum() if 'reviews' in month2_data.columns else 0

                    messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Provide a friendly and concise comparison of the total number of reviews for the {category_filter or 'all'} category "
                                f"between {month1} 2025 and {month2} 2025. The data shows {month1} 2025 had {month1_reviews} reviews, "
                                f"and {month2} 2025 had {month2_reviews} reviews. Only use the provided numbers and do not hallucinate data. "
                                f"Example: 'Hey! The total number of reviews for the {category_filter or 'all'} category in {month1} 2025 was {month1_reviews}, compared to {month2_reviews} in {month2} 2025!'"
                            )
                        }
                    ]
                    response = client.chat.completions.create(model="gpt-4o", messages=messages)
                    st.session_state.messages_insight.append({"role": "assistant", "content": response.choices[0].message.content})
                    with st.chat_message("assistant"):
                        st.write(response.choices[0].message.content)

                    with st.chat_message("assistant"):
                        st.write("### Analysis Results")
                        st.write(f"{month1} 2025: {month1_reviews} reviews")
                        st.write(f"{month2} 2025: {month2_reviews} reviews")

                    fig = go.Figure(data=[
                        go.Bar(x=[month1 + " 2025", month2 + " 2025"], y=[month1_reviews, month2_reviews], marker_color=['#FF6B6B', '#4ECDC4'])
                    ])
                    fig.update_layout(
                        title=f"Reviews Comparison - {category_filter.capitalize() if category_filter else 'All Categories'} ({month1} vs {month2})",
                        xaxis_title="Month",
                        yaxis_title="Number of Reviews",
                        height=500,
                        width=700
                    )
                    with st.chat_message("assistant"):
                        st.plotly_chart(fig)

            elif "total number of reviews per month" in prompt.lower():
                if 'reviews' not in df.columns:
                    st.warning("The 'reviews' column is not found in the uploaded data.")
                    st.stop()
                monthly_reviews = df_filtered.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()
                openai_data = monthly_reviews.to_string()
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Based on the provided data, provide a friendly and concise summary of the total number of reviews per month for the {category_filter or 'all'} category. "
                            f"Use the following grouped data with columns: {list(monthly_reviews.columns)}. "
                            f"Data:\n{openai_data}\n\n---\n\n {prompt} (e.g., 'Hey! The data shows a peak in January 2025 with 3000 reviews for {category_filter or 'all'}!')"
                        )
                    }
                ]
                response = client.chat.completions.create(model="gpt-4o", messages=messages)
                st.session_state.messages_insight.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)

                monthly_reviews = df_filtered.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()
                seen = set()
                unique_results = []
                for index, row in monthly_reviews.iterrows():
                    key = (row['month_year'], row['category'])
                    if key not in seen:
                        unique_results.append(row)
                        seen.add(key)
                monthly_reviews = pd.DataFrame(unique_results)

                with st.chat_message("assistant"):
                    st.write("### Analysis Results")
                    st.table(monthly_reviews.style.format({'reviews': '{:,.0f}'}))

                colors = {cat: '#FF6B6B' if i == 0 else '#4ECDC4' for i, cat in enumerate(monthly_reviews['category'].unique())}
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
                    title=f"Total Reviews Per Month by {category_filter.capitalize() if category_filter else 'All Categories'}",
                    xaxis_title="Month",
                    yaxis_title="Number of Reviews",
                    height=500,
                    width=700,
                    barmode='group',
                    showlegend=True
                )
                with st.chat_message("assistant"):
                    st.plotly_chart(fig)

            elif "reviews" in prompt.lower() and ("last month" in prompt.lower() or "this month" in prompt.lower()):
                if 'reviews' not in df.columns:
                    st.warning("The 'reviews' column is not found in the uploaded data.")
                    st.stop()
                current_date = datetime.now()
                current_month = current_date.month
                current_year = current_date.year
                last_month_year = current_year - 1 if current_month == 1 else current_year
                last_month = 12 if current_month == 1 else current_month - 1

                category = category_filter
                df_filtered = df[df['category'].str.lower().str.contains(category.lower(), na=False)] if category else df

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

                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Provide a friendly and concise comparison of the total number of reviews for the {category_filter or 'all'} category "
                            f"between last month and this month. The data shows last month had {last_month_reviews} reviews, "
                            f"and this month had {this_month_reviews} reviews. Only use the provided numbers and do not hallucinate data. "
                            f"Example: 'Hey! Last month had {last_month_reviews} reviews, while this month has {this_month_reviews} for the {category_filter or 'all'} category!'"
                        )
                    }
                ]
                response = client.chat.completions.create(model="gpt-4o", messages=messages)
                st.session_state.messages_insight.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)

                with st.chat_message("assistant"):
                    st.write("### Analysis Results")
                    st.write(f"This Month: {this_month_reviews} reviews")
                    st.write(f"Last Month: {last_month_reviews} reviews")

                fig = go.Figure(data=[
                    go.Bar(x=['Last Month', 'This Month'], y=[last_month_reviews, this_month_reviews], marker_color=['#FF6B6B', '#4ECDC4'])
                ])
                fig.update_layout(
                    title=f"Reviews Comparison - {category_filter if category_filter else 'All Categories'}",
                    xaxis_title="Period",
                    yaxis_title="Number of Reviews",
                    height=500,
                    width=700
                )
                with st.chat_message("assistant"):
                    st.plotly_chart(fig)

            elif any(word in prompt.lower() for word in ["most", "least"]):
                entity = "SKU" if "sku" in prompt.lower() else "product"
                metric = None
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ["sales", "sale"]):
                        metric = col
                        break
                if not metric:
                    metric = "reviews" if "reviews" in df.columns else None
                    if not metric:
                        st.warning("No 'sales' or 'reviews' column found in the uploaded data.")
                        st.stop()
                group_column = entity.lower() if entity.lower() in df.columns else "SKU"
                if group_column not in df.columns:
                    st.warning(f"Grouping column '{group_column}' not found in the dataset.")
                    st.stop()

                df['month_year'] = df['date'].dt.strftime('%B %Y')
                entity_metrics = df.groupby(['month_year', group_column])[metric].sum().reset_index()

                if entity_metrics.empty or entity_metrics[metric].isna().all():
                    st.warning(f"No valid {metric} data available for {entity}s.")
                    st.stop()

                with st.chat_message("assistant"):
                    st.write("### Analysis Results")
                    for month_year in entity_metrics['month_year'].unique():
                        month_data = entity_metrics[entity_metrics['month_year'] == month_year]
                        max_value = month_data[metric].max()
                        most_entities = month_data[month_data[metric] == max_value][group_column].tolist()
                        most_entities_str = ", ".join(most_entities) if len(most_entities) > 1 else most_entities[0]

                        min_value = month_data[month_data[metric] > 0][metric].min() if (month_data[metric] > 0).any() else 0
                        least_entities = month_data[month_data[metric] == min_value][group_column].tolist() if min_value > 0 else [None]
                        least_entities_str = ", ".join(filter(None, least_entities)) if len(least_entities) > 1 else (least_entities[0] if least_entities[0] else "None")

                        st.write(f"{month_year}: Most {metric}: {most_entities_str} ({max_value}), Least {metric}: {least_entities_str} ({min_value if min_value > 0 else 0})")

            else:
                messages = [
                    {
                        "role": "user",
                        "content": f"Provide a friendly response. I don’t fully understand your question about the data. Could you please ask about reviews, sales, or specific months? For example, 'What were the total number of reviews per month?' or 'Which SKU had the most sales?' Or upload a CSV file to start!"
                    }
                ]
                response = client.chat.completions.create(model="gpt-4o", messages=messages)
                st.session_state.messages_insight.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)

elif menu == "Shopify Catalog Analysis":
    st.title("🛒 Shopify Catalog Analysis")
    st.write("Chat with me about your Shopify catalog! Ask about stock levels or product updates.")

    # Display chat messages for Shopify Catalog Analysis
    for message in st.session_state.messages_shopify:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input for Shopify Catalog Analysis
    if prompt := st.chat_input("Ask me about your Shopify catalog! (e.g., 'Which products are out of stock, and how many?')"):
        st.session_state.messages_shopify.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Fetching Shopify catalog data via GraphQL..."):
            df = fetch_shopify_products()

        if df.empty:
            st.session_state.messages_shopify.append({"role": "assistant", "content": "Oops! I couldn’t fetch the Shopify data. Please check your API credentials."})
            with st.chat_message("assistant"):
                st.write("Oops! I couldn’t fetch the Shopify data. Please check your API credentials.")
        else:
            document = df.to_string()
            if "products are out of stock" in prompt.lower() and "how many" in prompt.lower():
                out_of_stock = df[df['inventory_quantity'] == 0]
                out_of_stock_count = len(out_of_stock)
                in_stock_count = len(df[df['inventory_quantity'] > 0])

                if out_of_stock_count > 0:
                    out_of_stock_list = out_of_stock[['title', 'sku']].drop_duplicates().to_dict('records')
                    sample_products = out_of_stock_list[:3]
                    sample_text = "\n".join([f"{i+1}. {item['title']} (SKU: {item['sku']}) - 0 items in stock" for i, item in enumerate(sample_products)])
                    if len(out_of_stock_list) > 3:
                        sample_text += "\n(and more!)"

                    messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Here's the Shopify catalog data: {document} \n\n---\n\n {prompt} Provide a single, user-friendly, concise, and precise response. "
                                f"There are {out_of_stock_count} products out of stock. Include the total count, list up to 3 examples with titles, SKUs, and quantities (0), "
                                f"and encourage restocking with a fun tone (e.g., 'Time to restock! Let us know if you need help!'). Do not repeat information or split into separate sections. "
                                f"Example: 'Hey there! We’ve got {out_of_stock_count} products out of stock, including: 1. Short-sleeve Tshirt 1 (SKU: 5) - 0 items in stock, "
                                f"2. Short-sleeve Tshirt (SKU: None) - 0 items in stock, 3. Short-sleeve Tshirt (SKU: 10) - 0 items in stock (and more!). Time to restock! Let us know if you need help!'"
                            )
                        }
                    ]
                    response = client.chat.completions.create(model="gpt-4o", messages=messages)
                    st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                    with st.chat_message("assistant"):
                        st.write(response.choices[0].message.content)
                    st.rerun()

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
                    st.session_state.messages_shopify.append({"role": "assistant", "content": fig})

                else:
                    messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Here's the Shopify catalog data: {document} \n\n---\n\n {prompt} Provide a single, user-friendly, concise, and precise response. "
                                f"Say something like 'Hey there! Great news—there are no products out of stock right now!'"
                            )
                        }
                    ]
                    response = client.chat.completions.create(model="gpt-4o", messages=messages)
                    st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                    with st.chat_message("assistant"):
                        st.write(response.choices[0].message.content)
                    st.rerun()

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
                    st.session_state.messages_shopify.append({"role": "assistant", "content": fig})

            elif "last month" in prompt.lower() and "this month" in prompt.lower():
                current_date = datetime.now()
                current_month = current_date.month
                current_year = current_date.year
                last_month_year = current_year - 1 if current_month == 1 else current_year
                last_month = 12 if current_month == 1 else current_month - 1

                category = "Electronics" if "electronics" in prompt.lower() else None
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

                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Here's the Shopify catalog data: {document} \n\n---\n\n {prompt} Provide a single, user-friendly, concise, and precise response. "
                            f"The data shows last month had {last_month_count} product updates, and this month has {this_month_count} product updates. "
                            f"Include the counts in the response without repeating information. "
                            f"Example: 'Hey! Last month saw {last_month_count} product updates, while this month has {this_month_count} for the {category or 'all'} category!'"
                        )
                    }
                ]
                response = client.chat.completions.create(model="gpt-4o", messages=messages)
                st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)
                st.rerun()

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
                st.session_state.messages_shopify.append({"role": "assistant", "content": fig})

            else:
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Here's the Shopify catalog data: {document} \n\n---\n\n {prompt} Provide a single, user-friendly, concise, and precise response. "
                            f"If the question is unclear, suggest options like 'Hey! You can ask me about stock levels (e.g., Which products are out of stock?) "
                            f"or product updates (e.g., How many products were updated last month?).'"
                        )
                    }
                ]
                response = client.chat.completions.create(model="gpt-4o", messages=messages)
                st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)
                st.rerun()
