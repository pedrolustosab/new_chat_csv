import os
import glob
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Function to initialize the LLM agent
def initialize_llm_agent(llm, dataframe, tools):
    prefix = """
        You are an expert data scientist with extensive knowledge of CSV operations and pandas dataframes.
        Utilize the PythonREPLTool for detailed analysis and to create visually appealing plots using matplotlib.pyplot.
        Ensure all charts are designed attractively with a pleasing color palette.
        Display exact values clarity.
    """
    
    agent = create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        prefix=prefix,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=tools,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )
    return agent

# Function to analyze the dataframe using the agent
def analyze_dataframe(agent, query):
    try:
        response = agent.invoke(query)
        return response.get("output", "No output received.")
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to display matplotlib charts in Streamlit
def display_chart():
    plt.tight_layout()
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.set_page_config(page_title="CSV Data Analysis with LLM", layout="wide", initial_sidebar_state="expanded")
    st.title("📊 CSV Data Analysis with LLM")
    st.subheader("Analyze your data with the power of GPT-4 and pandas 🧠")

    # Sidebar for OpenAI API Key input
    st.sidebar.header("🔑 Enter OpenAI API Key")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Input your OpenAI API key here.")
    
    if openai_api_key:
        st.sidebar.success("API key provided!")

    # Ensure the user has provided the API key
    if not openai_api_key:
        st.error("Please provide your OpenAI API key in the sidebar to continue.")
        return

    # Sidebar for CSV Upload and Model Information
    st.sidebar.header("📁 Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Select a CSV file", type=["csv"])

    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ How to use")
    st.sidebar.write("1. Enter your OpenAI API key.")
    st.sidebar.write("2. Upload your CSV file.")
    st.sidebar.write("3. Ask questions about the data.")
    st.sidebar.write("4. View generated charts and results.")

    # Columns layout for better organization
    col1, col2 = st.columns([3, 2])

    if uploaded_file is not None:
        with col1:
            df = pd.read_csv(uploaded_file)
            st.write("### 📄 Data Preview")
            st.dataframe(df.head(), height=300)
            st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")

        # Initialize the LLM and tools
        llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key)
        tools = [PythonREPLTool()]
        agent = initialize_llm_agent(llm, df, tools)

        with col2:
            # Get user's query
            st.write("### 🔍 Ask a Question")
            query = st.text_area("Ask a question related to this data (e.g., 'Show me a summary of this data.')")
            st.button("Submit")
            
            if query:
                with st.spinner("Analyzing data..."):
                    result = analyze_dataframe(agent, query)
                    st.success("Analysis complete!")
                    st.write(result)
                    
                # Display generated charts directly from matplotlib
                display_chart()
                    
    else:
        with col1:
            st.info("Upload a CSV file to start the analysis.")

    # Adding footer with contact information
    st.markdown(
        """
        <style>
        footer {visibility: hidden;}
        footer:after {
            content:'Powered by Streamlit and GPT-4'; 
            visibility: visible;
            display: block;
            position: relative;
            color: #4CAF50;
            padding: 10px;
            text-align: center;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
