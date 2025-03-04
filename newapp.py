import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from datetime import datetime

# Access OpenAI API key securely from Streamlit secrets, with fallback for local testing
try:
    openai_api_key = st.secrets.get("openai.api_key")
    if not openai_api_key:
        st.warning("OpenAI API key not found in secrets. Using hardcoded key for local testing (remove for deployment).")
        openai_api_key = "your_openai_api_key_here"  # Replace with your key for local testing only
    openai.api_key = openai_api_key
except (KeyError, AttributeError, FileNotFoundError) as e:
    st.error(f"OpenAI configuration error: {str(e)}. Please configure 'openai.api_key' in Streamlit Cloud under 'Manage app' > 'Secrets' or use a local secrets.toml file.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Excel Data Chat", layout="wide")

# Custom CSS for chat styling
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .user-message { background-color: #f0f2f6; }
    .bot-message { background-color: #e8f5e9; }
    </style>
""", unsafe_allow_html=True)

# Load or upload Excel file
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def load_excel_data(uploaded_file=None):
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading uploaded Excel file: {str(e)}")
            return pd.DataFrame()
    else:
        st.error("Please upload an Excel file to analyze.")
        return pd.DataFrame()

def generate_nlp_response(conversation_text, context=None, data=None):
    """Generate an NLP response and graph instruction using OpenAI's GPT API"""
    try:
        # Prepare the prompt with context, data summary, and user query
        prompt = f"User query: {conversation_text}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        if data is not None and not data.empty:
            prompt += f"Available data columns: {list(data.columns)}\nData summary: {data.describe().to_string()}\n\n"
        prompt += "Provide a natural, concise response as a helpful AI assistant for analyzing data from an uploaded Excel file. If the user asks about a top-performing product, returns, or specific metrics, suggest insights, calculations, or visualizations based on the data. Return the response in this format: 'Response: [your response]' and if a graph is needed, include 'Graph: [description of the graph, e.g., bar chart of top products by performance]'."

        # Call OpenAI API with the new chat completions endpoint
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" for more advanced responses if available
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for analyzing data from an uploaded Excel file and providing insights or visualizations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  # Increased for more detailed responses
            temperature=0.7  # Balanced between creativity and coherence
        )
        
        full_response = response.choices[0].message.content.strip()
        # Parse the response for text and graph instructions
        if "Response:" in full_response and "Graph:" in full_response:
            response_text = full_response.split("Response:")[1].split("Graph:")[0].strip()
            graph_instruction = full_response.split("Graph:")[1].strip()
            return response_text, graph_instruction
        return full_response, None
    except Exception as e:
        st.error(f"Error generating NLP response: {str(e)}")
        return "Sorry, I encountered an issue processing your request.", None

def create_graph(data, graph_instruction):
    """Create a graph based on the OpenAI instruction"""
    try:
        if "bar chart" in graph_instruction.lower():
            if "top products" in graph_instruction.lower() and "performance" in graph_instruction.lower():
                top_products = data.sort_values(by="Performance", ascending=False).head(5)  # Adjust column name as needed
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.barplot(data=top_products, x="Product", y="Performance", ax=ax)  # Adjust column names
                ax.set_title("Top 5 Products by Performance")
                ax.set_xlabel("Product")
                ax.set_ylabel("Performance")
                return fig
            elif "returns" in graph_instruction.lower():
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.barplot(data=data, x="Product", y="Returns", ax=ax)  # Adjust column names
                ax.set_title("Returns by Product")
                ax.set_xlabel("Product")
                ax.set_ylabel("Returns")
                return fig
        elif "line chart" in graph_instruction.lower():
            if "performance over time" in graph_instruction.lower():
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.lineplot(data=data, x="Date", y="Performance", ax=ax)  # Adjust column names
                ax.set_title("Performance Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Performance")
                return fig
        return None
    except Exception as e:
        st.error(f"Error creating graph: {str(e)}")
        return None

def process_conversation(conversation_text, context=None, data=None):
    """Process the user prompt, analyze Excel data, and return responses, calculations, and graphs"""
    try:
        if data is None or data.empty:
            st.error("No data available. Please upload an Excel file to analyze.")
            return {"response": "No data available to process.", "graphs": [], "insights": []}

        # Generate NLP response and graph instruction
        nlp_response, graph_instruction = generate_nlp_response(conversation_text, context, data)
        
        # Initialize analysis
        analysis = {"response": nlp_response, "graphs": []}
        
        # Simple keyword-based analysis for insights (can be enhanced with NLP)
        lines = conversation_text.strip().split('\n')
        insights = []
        for line in lines:
            if "top performing product" in line.lower():
                top_product = data.loc[data["Performance"].idxmax()]  # Adjust column name
                insights.append(f"Top-performing product: {top_product['Product']} with performance {top_product['Performance']}")
            elif "drove most return" in line.lower() or "most returns" in line.lower():
                top_return = data.loc[data["Returns"].idxmax()]  # Adjust column name
                insights.append(f"Product with most returns: {top_return['Product']} with returns {top_return['Returns']}")
            elif "average performance" in line.lower():
                avg_performance = data["Performance"].mean()  # Adjust column name
                insights.append(f"Average performance across products: {avg_performance:.2f}")
            elif "total returns" in line.lower():
                total_returns = data["Returns"].sum()  # Adjust column name
                insights.append(f"Total returns across all products: {total_returns}")

        # Create graph if instructed
        if graph_instruction:
            graph = create_graph(data, graph_instruction)
            if graph:
                analysis["graphs"].append(graph)
        
        analysis["insights"] = insights
        return analysis
    except Exception as e:
        st.error(f"Error in process_conversation: {str(e)}")
        return {"response": "Sorry, I encountered an error processing your request.", "graphs": [], "insights": []}

def main():
    st.title("Excel Data Chat")

    # Initialize session state for conversation history and uploaded data
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'excel_data' not in st.session_state:
        st.session_state.excel_data = None

    # File upload for Excel
    st.subheader("Upload Your Excel File")
    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

    if uploaded_file:
        st.session_state.excel_data = load_excel_data(uploaded_file)
        st.success("Excel file uploaded successfully!")

    # Chat window (only active if data is uploaded)
    if st.session_state.excel_data is not None and not st.session_state.excel_data.empty:
        st.subheader("Ask about your data")
        conversation = st.text_area("Type your query (e.g., 'Show me the top-performing product,' 'Graph returns over time,' 'What’s the average performance?')", height=200)

        if st.button("Send"):
            if conversation:
                with st.spinner("Analyzing your query..."):
                    try:
                        # Add user input to conversation history
                        st.session_state.conversation_history.append({"role": "user", "content": conversation})

                        # Process the conversation with context from history
                        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
                        analysis = process_conversation(conversation, context, st.session_state.excel_data)

                        # Display the system's conversational response
                        if analysis["response"]:
                            st.markdown(f'<div class="stChatMessage bot-message">Bot: {analysis["response"]}</div>', unsafe_allow_html=True)
                            st.session_state.conversation_history.append({"role": "bot", "content": analysis["response"]})

                        # Display conversation history
                        st.subheader("Conversation History")
                        for msg in st.session_state.conversation_history:
                            role_class = "user-message" if msg["role"] == "user" else "bot-message"
                            st.markdown(f'<div class="stChatMessage {role_class}">{"You" if msg["role"] == "user" else "Bot"}: {msg["content"]}</div>', unsafe_allow_html=True)

                        # Display insights (answers/calculations)
                        if analysis["insights"]:
                            st.subheader("Answers & Calculations")
                            for insight in analysis["insights"]:
                                st.write(f"- {insight}")

                        # Display graphs
                        if analysis["graphs"]:
                            st.subheader("Visualizations")
                            for graph in analysis["graphs"]:
                                st.pyplot(graph)
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            else:
                st.error("Please enter a query to analyze")
    else:
        st.warning("Please upload an Excel file to start analyzing data.")

if __name__ == "__main__":
    main()