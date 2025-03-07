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

# Initialize session state at the top to avoid TypeError
if "messages_csv" not in st.session_state:
    st.session_state.messages_csv = []
if "messages_shopify" not in st.session_state:
    st.session_state.messages_shopify = []
if "df_csv" not in st.session_state:
    st.session_state.df_csv = None  # Start with no data until uploaded
if "last_processed_prompt_csv" not in st.session_state:
    st.session_state.last_processed_prompt_csv = None
if "last_processed_prompt_shopify" not in st.session_state:
    st.session_state.last_processed_prompt_shopify = None
if "uploader_at_bottom" not in st.session_state:
    st.session_state.uploader_at_bottom = False  # Track uploader position

# Custom CSS with conditional padding based on uploader position
if st.session_state.uploader_at_bottom:
    st.markdown(
        """
        <style>
        /* Ensure the app takes full height and uses Flexbox */
        .stApp {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        /* Main content area should be scrollable with padding for footer */
        .content {
            flex: 1;
            overflow-y: auto;
            padding-bottom: 120px; /* Space for fixed footer */
            box-sizing: border-box;
        }
        /* Fixed footer at the bottom */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f0f2f6;
            padding: 10px;
            z-index: 1000;
            width: 100%;
            display: block; /* Show footer when uploader is at bottom */
        }
        /* Reduced space between uploader and chat input */
        .stFileUploader {
            margin-bottom: 5px;
        }
        /* Ensure chat input aligns properly */
        .stChatInput {
            margin-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        /* Ensure the app takes full height and uses Flexbox */
        .stApp {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        /* Main content area should be scrollable with no padding */
        .content {
            flex: 1;
            overflow-y: auto;
            padding-bottom: 0px; /* No padding when uploader is at top */
            box-sizing: border-box;
        }
        /* Fixed footer at the bottom */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f0f2f6;
            padding: 10px;
            z-index: 1000;
            width: 100%;
            display: none; /* Hidden by default, shown when uploader moves */
        }
        /* Reduced space between uploader and chat input */
        .stFileUploader {
            margin-bottom: 5px;
        }
        /* Ensure chat input aligns properly */
        .stChatInput {
            margin-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

# Add sidebar with menu items
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["CSV Analysis", "Shopify Catalog Analysis"])

# CSV Analysis tab
if menu == "CSV Analysis":
    st.title("📄 CSV Analysis")
    st.write("Chat with me about your data! Upload a CSV and ask about reviews, sales, or specific months.")

    # File uploader at top by default
    if not st.session_state.uploader_at_bottom:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_uploader", help="Upload your data file to analyze.")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.lower()  # Normalize column names
            st.session_state.df_csv = df
            # No success message appended as per request

    # Response area with padding
    st.markdown('<div class="content">', unsafe_allow_html=True)
    response_container = st.container()
    with response_container:
        if not st.session_state.messages_csv:
            pass  # Removed "Responses will appear here."
        for idx, message in enumerate(st.session_state.messages_csv):
            with st.chat_message(message["role"]):
                if isinstance(message["content"], go.Figure):
                    st.plotly_chart(message["content"], key=f"plotly_chart_{idx}")
                else:
                    # Handle all string content (including HTML tables) directly
                    st.markdown(message["content"], unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Fixed footer (shown after first response)
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    if st.session_state.uploader_at_bottom:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_uploader", help="Upload your data file to analyze.")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.lower()  # Normalize column names
            st.session_state.df_csv = df
            # No success message appended as per request
    if prompt := st.chat_input("Ask me about your data! (e.g., 'What were the total number of reviews per month?')"):
        # Append user prompt
        st.session_state.messages_csv.append({"role": "user", "content": prompt})

        # Load and process data
        df = st.session_state.df_csv
        if df is None or df.empty:
            st.session_state.messages_csv.append({"role": "assistant", "content": "No data available. Please upload a CSV file to analyze."})
            st.rerun()
        else:
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
            if df['date'].isna().all():
                st.session_state.messages_csv.append({"role": "assistant", "content": "No valid dates found in the 'date' column. Please ensure dates are in DD/MM/YYYY format."})
                st.rerun()
            else:
                # Process the query only if it's different from the last processed prompt
                if st.session_state.last_processed_prompt_csv != prompt:
                    st.session_state.last_processed_prompt_csv = prompt

                    df['month_year'] = df['date'].dt.strftime('%B %Y')
                    df['category'] = df['category'].str.lower().replace("tootbrush", "toothbrush")

                    category_filter = None
                    if "toothbrush" in prompt.lower():
                        category_filter = "toothbrush"
                    elif "all categories" in prompt.lower() or "all" in prompt.lower():
                        category_filter = None
                    df_filtered = df if category_filter is None else df[df['category'] == category_filter]

                    # Process the query
                    if "total number of reviews per month" in prompt.lower() or "reviews per month for all categories" in prompt.lower():
                        monthly_reviews = df_filtered.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()
                        openai_data = monthly_reviews.to_string()
                        messages = [
                            {
                                "role": "user",
                                "content": (
                                    f"Based on the provided data, provide a friendly and concise summary of the total number of reviews per month for all categories. "
                                    f"Use the following grouped data with columns: {list(monthly_reviews.columns)}. "
                                    f"Data:\n{openai_data}\n\n---\n\n {prompt}"
                                )
                            }
                        ]
                        response = client.chat.completions.create(model="gpt-4o", messages=messages)
                        st.session_state.messages_csv.append({"role": "assistant", "content": response.choices[0].message.content})

                        monthly_reviews = df_filtered.groupby(['month_year', 'category'], as_index=False)['reviews'].sum()
                        seen = set()
                        unique_results = []
                        for index, row in monthly_reviews.iterrows():
                            key = (row['month_year'], row['category'])
                            if key not in seen:
                                unique_results.append(row)
                                seen.add(key)
                        monthly_reviews = pd.DataFrame(unique_results)
                        st.session_state.messages_csv.append({"role": "assistant", "content": monthly_reviews.style.format({'reviews': '{:,.0f}'}).to_html()})  # Removed title

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
                            title=f"Total Reviews Per Month by {'Toothbrush' if category_filter == 'toothbrush' else 'Category'}",
                            xaxis_title="Month",
                            yaxis_title="Number of Reviews",
                            height=500,
                            width=700,
                            barmode='group',
                            showlegend=True
                        )
                        st.session_state.messages_csv.append({"role": "assistant", "content": fig})

                        # Move uploader to bottom after first assistant response
                        assistant_messages = [msg for msg in st.session_state.messages_csv if msg["role"] == "assistant"]
                        if len(assistant_messages) >= 1 and not st.session_state.uploader_at_bottom:
                            st.session_state.uploader_at_bottom = True
                            st.rerun()

                    elif "compare reviews" in prompt.lower() and any(month in prompt.lower() for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
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
                                        f"and {month2} 2025 had {month2_reviews} reviews."
                                    )
                                }
                            ]
                            response = client.chat.completions.create(model="gpt-4o", messages=messages)
                            st.session_state.messages_csv.append({"role": "assistant", "content": response.choices[0].message.content})

                            st.session_state.messages_csv.append({"role": "assistant", "content": f"{month1} 2025: {month1_reviews} reviews\n{month2} 2025: {month2_reviews} reviews"})  # Removed title

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
                            st.session_state.messages_csv.append({"role": "assistant", "content": fig})

                            # Move uploader to bottom after first assistant response
                            assistant_messages = [msg for msg in st.session_state.messages_csv if msg["role"] == "assistant"]
                            if len(assistant_messages) >= 1 and not st.session_state.uploader_at_bottom:
                                st.session_state.uploader_at_bottom = True
                                st.rerun()

                    elif "reviews" in prompt.lower() and ("last month" in prompt.lower() or "this month" in prompt.lower()):
                        current_date = datetime.now()
                        current_month = current_date.month
                        current_year = current_date.year
                        last_month_year = current_year - 1 if current_month == 1 else current_year
                        last_month = 12 if current_month == 1 else current_month - 1

                        category = "toothbrush" if "toothbrush" in prompt.lower() else None
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

                        messages = [
                            {
                                "role": "user",
                                "content": (
                                    f"Provide a friendly and concise comparison of the total number of reviews for the {category_filter or 'all'} category "
                                    f"between last month and this month. The data shows last month had {last_month_reviews} reviews, "
                                    f"and this month had {this_month_reviews} reviews."
                                )
                            }
                        ]
                        response = client.chat.completions.create(model="gpt-4o", messages=messages)
                        st.session_state.messages_csv.append({"role": "assistant", "content": response.choices[0].message.content})

                        st.session_state.messages_csv.append({"role": "assistant", "content": f"This Month: {this_month_reviews} reviews\nLast Month: {last_month_reviews} reviews"})  # Removed title

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
                        st.session_state.messages_csv.append({"role": "assistant", "content": fig})

                        # Move uploader to bottom after first assistant response
                        assistant_messages = [msg for msg in st.session_state.messages_csv if msg["role"] == "assistant"]
                        if len(assistant_messages) >= 1 and not st.session_state.uploader_at_bottom:
                            st.session_state.uploader_at_bottom = True
                            st.rerun()

                    elif any(word in prompt.lower() for word in ["most", "least"]) and any(metric in prompt.lower() for metric in ["reviews", "sales", "sale"]):
                        entity = "sku" if "sku" in prompt.lower() else "product"
                        # Map the metric to the correct column name (normalized to lowercase)
                        metric = "sales" if "sale" in prompt.lower() or "sales" in prompt.lower() else "reviews"
                        group_column = entity
                        if group_column not in df.columns:
                            st.session_state.messages_csv.append({"role": "assistant", "content": f"Grouping column '{group_column}' not found in the dataset."})
                        else:
                            df['month_year'] = df['date'].dt.strftime('%B %Y')
                            entity_metrics = df.groupby(['month_year', group_column])[metric].sum().reset_index()

                            if entity_metrics.empty or entity_metrics[metric].isna().all():
                                st.session_state.messages_csv.append({"role": "assistant", "content": f"No valid {metric} data available for {entity}s."})
                            else:
                                result_text = ""
                                for month_year in entity_metrics['month_year'].unique():
                                    month_data = entity_metrics[entity_metrics['month_year'] == month_year]
                                    max_value = month_data[metric].max()
                                    most_entities = month_data[month_data[metric] == max_value][group_column].tolist()
                                    most_entities_str = ", ".join(map(str, most_entities)) if len(most_entities) > 1 else str(most_entities[0])

                                    min_value = month_data[month_data[metric] > 0][metric].min() if (month_data[metric] > 0).any() else 0
                                    least_entities = month_data[month_data[metric] == min_value][group_column].tolist() if min_value > 0 else [None]
                                    least_entities_str = ", ".join(filter(None, map(str, least_entities))) if len(least_entities) > 1 else (str(least_entities[0]) if least_entities[0] else "None")

                                    result_text += f"{month_year}: Most {metric}: {most_entities_str} ({max_value}), Least {metric}: {least_entities_str} ({min_value if min_value > 0 else 0})\n"
                                st.session_state.messages_csv.append({"role": "assistant", "content": result_text})  # Removed title

                                # Move uploader to bottom after first assistant response
                                assistant_messages = [msg for msg in st.session_state.messages_csv if msg["role"] == "assistant"]
                                if len(assistant_messages) >= 1 and not st.session_state.uploader_at_bottom:
                                    st.session_state.uploader_at_bottom = True
                                    st.rerun()

                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": (
                                    f"Based on the provided data, please interpret and respond to the following query in a friendly and concise manner: {prompt}. "
                                    f"If the query is unclear or no data is available, suggest uploading a CSV file and provide alternatives such as 'What were the total number of reviews per month?', "
                                    f"'Compare reviews for January and February', or 'Which SKU had the most sales?'"
                                )
                            }
                        ]
                        response = client.chat.completions.create(model="gpt-4o", messages=messages)
                        st.session_state.messages_csv.append({"role": "assistant", "content": response.choices[0].message.content})

                        # Move uploader to bottom after first assistant response
                        assistant_messages = [msg for msg in st.session_state.messages_csv if msg["role"] == "assistant"]
                        if len(assistant_messages) >= 1 and not st.session_state.uploader_at_bottom:
                            st.session_state.uploader_at_bottom = True
                            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Shopify Catalog Analysis
elif menu == "Shopify Catalog Analysis":
    st.title("🛒 Shopify Catalog Analysis")
    st.write("Analyze your Shopify catalog data. Query stock levels, product updates, or other metrics as needed.")

    # Response area (scrollable)
    st.markdown('<div class="content">', unsafe_allow_html=True)
    response_container = st.container()
    with response_container:
        if not st.session_state.messages_shopify:
            pass  # No placeholder text
        for idx, message in enumerate(st.session_state.messages_shopify):
            with st.chat_message(message["role"]):
                if isinstance(message["content"], go.Figure):
                    st.plotly_chart(message["content"], key=f"shopify_plotly_chart_{idx}")
                else:
                    st.write(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Fixed footer for chat input
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    if prompt := st.chat_input("Query your Shopify catalog (e.g., 'Which products are out of stock, and how many?')"):
        # Append user prompt
        st.session_state.messages_shopify.append({"role": "user", "content": prompt})

        with st.spinner("Fetching Shopify catalog data via GraphQL..."):
            df = fetch_shopify_products()

        if df.empty:
            st.session_state.messages_shopify.append({"role": "assistant", "content": "Unable to retrieve Shopify catalog data. Please verify your API credentials and ensure connectivity."})
            st.rerun()
        else:
            # Process the query only if it's different from the last processed prompt
            if st.session_state.last_processed_prompt_shopify != prompt:
                st.session_state.last_processed_prompt_shopify = prompt

                document = df.to_string()
                if "products are out of stock" in prompt.lower() and "how many" in prompt.lower():
                    out_of_stock = df[df['inventory_quantity'] == 0]
                    out_of_stock_count = len(out_of_stock)
                    in_stock_count = len(df[df['inventory_quantity'] > 0])

                    if out_of_stock_count > 0:
                        out_of_stock_list = out_of_stock[['title', 'sku']].drop_duplicates().to_dict('records')
                        sample_products = out_of_stock_list[:3]
                        sample_text = "\n".join([f"{i+1}. {item['title']} (SKU: {item['sku']}) - 0 units in stock" for i, item in enumerate(sample_products)])
                        if len(out_of_stock_list) > 3:
                            sample_text += "\n(Additional products not listed.)"

                        messages = [
                            {
                                "role": "user",
                                "content": (
                                    f"Here is the Shopify catalog data: {document} \n\n---\n\n {prompt} Provide a professional, concise, and precise response. "
                                    f"There are {out_of_stock_count} products out of stock. Include the total count, list up to 3 examples with titles, SKUs, and quantities (0), "
                                    f"and recommend reviewing inventory levels."
                                )
                            }
                        ]
                        response = client.chat.completions.create(model="gpt-4o", messages=messages)
                        st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                        with st.chat_message("assistant"):
                            st.write(response.choices[0].message.content)

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
                        st.rerun()

                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": (
                                    f"Here is the Shopify catalog data: {document} \n\n---\n\n {prompt} Provide a professional, concise, and precise response. "
                                    f"Indicate that there are currently no products out of stock."
                                )
                            }
                        ]
                        response = client.chat.completions.create(model="gpt-4o", messages=messages)
                        st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                        with st.chat_message("assistant"):
                            st.write(response.choices[0].message.content)

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
                        st.rerun()

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
                                f"Here is the Shopify catalog data: {document} \n\n---\n\n {prompt} Provide a professional, concise, and precise response. "
                                f"The data indicates that last month had {last_month_count} product updates, and this month has {this_month_count} product updates."
                            )
                        }
                    ]
                    response = client.chat.completions.create(model="gpt-4o", messages=messages)
                    st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                    with st.chat_message("assistant"):
                        st.write(response.choices[0].message.content)

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
                    st.rerun()

                else:
                    messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Here is the Shopify catalog data: {document} \n\n---\n\n {prompt} Provide a professional, concise, and precise response. "
                                f"If the query is unclear, suggest alternatives such as 'You may query stock levels (e.g., Which products are out of stock?) "
                                f"or product updates (e.g., How many products were updated last month?).'"
                            )
                        }
                    ]
                    response = client.chat.completions.create(model="gpt-4o", messages=messages)
                    st.session_state.messages_shopify.append({"role": "assistant", "content": response.choices[0].message.content})
                    with st.chat_message("assistant"):
                        st.write(response.choices[0].message.content)
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
