# CSV Data Analysis with LLM

This Streamlit application allows users to analyze CSV data using the power of GPT-4 and pandas. It provides an intuitive interface for uploading CSV files, asking questions about the data, and viewing generated charts and results.

## Features

- CSV file upload
- Data preview with basic statistics
- Natural language querying of the data
- Automatic chart generation based on queries
- Powered by GPT-4 for intelligent data analysis

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/csv-data-analysis-llm.git
   cd csv-data-analysis-llm
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key to the file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run main.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the sidebar to upload a CSV file.

4. Once a file is uploaded, you can see a preview of the data and ask questions about it in natural language.

5. The app will analyze your query using GPT-4 and provide results, including any relevant charts or graphs.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Matplotlib
- Langchain
- OpenAI API key

For a full list of dependencies, see `requirements.txt`.


