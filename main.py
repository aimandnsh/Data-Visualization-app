import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os


#load .env file
load_dotenv()

# Initialize OpenAI api client
openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to load data
def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None 

# Function to display dataset information
def display_dataset_info(data):
    st.subheader('Original Data')
    st.write(data)

    with st.sidebar.expander("Dataset Information"):
        st.write(f"Number of Rows: {data.shape[0]}")
        st.write(f"Number of Columns: {data.shape[1]}")
    
    # Check for NaN values
    nan_values = data.isnull().sum()
    with st.sidebar.expander("NaN Values"):
        for column, nan_count in nan_values.items():
            st.write(f"{column}: {nan_count}")

# Function to display dataset description
def display_dataset_description(data):
    with st.sidebar.expander("Dataset Description"):
        st.write(data.describe())

# Function to display unique count of values in a column
def display_unique_count(data):
    with st.sidebar.expander("Unique Count Function"):
        selected_column = st.selectbox("Select a column", data.columns)
        if selected_column:
            unique_values_count = data[selected_column].nunique()
            st.write(f"Unique Count of {selected_column}: {unique_values_count}")

# Function to display interactive table
def display_interactive_table(data):
    st.subheader('Filter Data')
    st.write(data)

# Function to apply filters to the dataset
def apply_filters(data):
    # Example filter: numeric columns
    with st.sidebar.expander("Apply Filter"):
        columns = data.columns
        selected_column = st.selectbox("Select a column to filter", columns)
        if selected_column:
            if data[selected_column].dtype == 'object':
                unique_values = data[selected_column].unique()
                selected_value = st.selectbox(f"Select value(s) for {selected_column}", ['All'] + list(unique_values))
                if selected_value != 'All':
                    filtered_data = data[data[selected_column] == selected_value]
                    return filtered_data
                else:
                    return data
            else:
                min_value = st.number_input(f"Minimum value of {selected_column}", min_value=data[selected_column].min(), max_value=data[selected_column].max())
                max_value = st.number_input(f"Maximum value of {selected_column}", min_value=data[selected_column].min(), max_value=data[selected_column].max(), value=data[selected_column].max())
                filtered_data = data[(data[selected_column] >= min_value) & (data[selected_column] <= max_value)]
                return filtered_data
        else:
            return data

# Function to display additional visualizations
def display_additional_visualizations(data):
    st.subheader('Additional Visualizations')

    # Choose a visualization type
    visualization_type = st.selectbox('Choose a visualization type', ['Histogram', 'Box Plot', 'Violin Plot', 'Line Plot', 'Scatter Plot', 'Area Chart', 'Correlation Analysis'])

    # Depending on the choice, display the corresponding visualization
    if visualization_type == 'Histogram':
        # Plot histogram using Plotly
        column_to_visualize = st.selectbox('Choose a column to visualize', ['None'] + list(data.columns))
        if column_to_visualize != 'None':
            bin_size = st.slider("Select bin size", min_value=1, max_value=100, value=10)
            color = st.color_picker('Pick a color for the histogram', '#00f900')
            fig = px.histogram(data, x=column_to_visualize, nbins=bin_size, color_discrete_sequence=[color])
            st.plotly_chart(fig)

    elif visualization_type in ['Box Plot', 'Violin Plot']:
        # Plot box plot or violin plot using Plotly
        selected_columns = st.multiselect("Select columns for visualization", data.columns)
        if selected_columns:
            colors = px.colors.qualitative.Plotly[:len(selected_columns)]
            if visualization_type == 'Box Plot':
                fig = px.box(data, y=selected_columns, color_discrete_sequence=colors)
            elif visualization_type == 'Violin Plot':
                fig = px.violin(data, y=selected_columns, color_discrete_sequence=colors)
            st.plotly_chart(fig)
            
    elif visualization_type == 'Line Plot':
        # Choose X-axis column
        x_column = st.selectbox('Choose the X-axis column', ['None'] + list(data.columns))
        # Choose Y-axis column(s)
        y_columns = st.multiselect('Choose the Y-axis column(s)', data.columns)
        # Choose colors for the line plot (one color per selected column)
        colors = [st.color_picker(f"Pick a color for {column}", "#00f900") for column in y_columns]
        # If both X-axis and at least one Y-axis column are selected
        if x_column != 'None' and y_columns:
            fig = px.line(data, x=x_column, y=y_columns, color_discrete_sequence=colors)
            st.plotly_chart(fig)

    elif visualization_type == 'Scatter Plot':
        # Plot scatter plot using Plotly
        x_columns = st.multiselect('Choose the X-axis column(s)', data.columns)
        y_columns = st.multiselect('Choose the Y-axis column(s)', data.columns)
    
        if x_columns and y_columns:
            fig = px.scatter()
            colors = px.colors.qualitative.Alphabet
            color_idx = 0
            for x_col in x_columns:
                for y_col in y_columns:
                    scatter = px.scatter(data, x=data[x_col], y=data[y_col], labels={x_col: x_col, y_col: y_col})
                    scatter.update_traces(marker=dict(color=colors[color_idx]), name=f'{x_col} vs {y_col}')
                    fig.add_trace(scatter['data'][0])
                    color_idx = (color_idx + 2) % len(colors)
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig)

    elif visualization_type == 'Area Chart':
        # Plot area chart using Plotly
        x_column = st.selectbox('Choose the X-axis column', ['None'] + list(data.columns))
        y_columns = st.multiselect('Choose the Y-axis column(s)', data.columns)
        if x_column != 'None' and y_columns:
            fig = px.area(data, x=x_column, y=y_columns)
            st.plotly_chart(fig)

    elif visualization_type == 'Correlation Analysis':
        # Select only numeric columns from the dataframe
        numeric_data = data.select_dtypes(include='number')
        
        # Plot correlation heatmap using seaborn
        corr = numeric_data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
@st.cache_data        
def openai_text(data, query):
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0.3), data, verbose=True)
    answer = agent.run(query)
    
    return answer


# Main function
def main():
    st.set_page_config(
        page_title='Data Friendly',
        page_icon='📊',
        layout='wide')
    st.title('Data Exploration and Visualization App 📊')
    st.subheader("Please upload your CSV file to explore and visualize your data. Hope this app can provide your needs 😀")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        #Load data
        data = load_data(uploaded_file)
        
        col1, col2 = st.columns([1,1])
        
        if data is not None:
            with col1:
                # Display dataset information
                display_dataset_info(data)
                
                # Display dataset description
                display_dataset_description(data)
                # Apply filters
                data = apply_filters(data)
                
                # Display interactive table
                display_interactive_table(data)

                # Display unique count of values
                display_unique_count(data)

                
            with col2:
                # Display chatbox to ask openai
                st.subheader("Chat with your uploaded file")
                st.write("Ask this bot to help you anlyze the dataset")
                query = st.text_input("Enter Query: ")
                if st.button("Generate"):
                    st.write(openai_text(data, query))
                    
                display_additional_visualizations(data)

if __name__ == '__main__':
    main()
