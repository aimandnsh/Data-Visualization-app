# Data-Visualization-app

![image](https://github.com/aimandnsh/Data-Visualization-app/assets/150990001/628062c7-5f6b-4e65-975c-1306a0eec9c4)

# DataVision Hub 📊

DataVision Hub is a web application built with Streamlit that allows users to explore and visualize their datasets easily. With DataVision Hub, users can upload their CSV files, apply filters, visualize data in various formats, and even interact with an AI-powered chatbot to gain insights from their data.

## Features

- **Data Upload**: Easily upload CSV files to analyze and visualize.
- **Data Exploration**: Explore dataset information, including the number of rows and columns, and check for NaN values.
- **Filtering**: Apply filters to the dataset based on column values or numeric ranges.
- **Interactive Visualizations**: Visualize data using histograms, box plots, violin plots, line plots, scatter plots, area charts, and correlation analysis.
- **Chatbot Integration**: Interact with an AI chatbot powered by OpenAI to ask questions and gain insights from the dataset.
  
## Usage

1. **Upload CSV File**: Click on the "Upload a CSV file" button to upload your dataset.
2. **Explore Data**: Once the file is uploaded, explore dataset information, apply filters, and visualize data using the options provided.
3. **Chat with Chatbot**: Enter your query in the text box provided to interact with the AI chatbot. Click on "Generate" to get responses from the chatbot.
4. **Visualize Data**: Choose different visualization types from the dropdown menu and customize the visualization parameters as needed.
5. **Download Results**: Optionally, download the filtered dataset or visualizations for further analysis.

## Installation

To run DataVision Hub locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/aimandnsh/datavision-hub.git
   ```

2. Navigate to the project directory:

   ```bash
   cd datavision-hub
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:

   - Create a `.env` file in the project directory.
   - Add your OpenAI API key to the `.env` file:

     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

5. Run the Streamlit app:

   ```bash
   streamlit run main.py
   ```

## Dependencies

DataVision Hub is built with the following dependencies:

- Streamlit
- Pandas
- Plotly Express
- Matplotlib
- Seaborn
- OpenAI
- dotenv

For detailed information on dependencies and versions, refer to the `requirements.txt` file.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
