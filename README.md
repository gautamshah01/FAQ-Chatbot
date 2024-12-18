
# CodeBasics FAQ Chatbot

This project implements a chatbot using `LangChain`, `FAISS`, and `Streamlit` for an FAQ-based question-answering system, powered by a knowledge base stored in CSV format. The chatbot is designed for an EdTech platform, answering questions related to courses, tutorials, and other FAQs.

![image](https://github.com/user-attachments/assets/26ace9ab-00a6-49cc-835c-2be90fc5c438)



## Features

- **FAQ-based Q&A**: Users can ask questions, and the chatbot provides answers based on a pre-trained vector database built from a CSV of questions and answers.
- **Streamlit UI**: A user-friendly interface to interact with the chatbot.
- **Error Handling**: Handles errors in vector database creation, CSV loading, and during question processing.
- **Knowledge Base**: The knowledge base is stored in a vector database, using FAISS for fast similarity search.

## Project Structure

The project consists of two main Python files:

1. **`main.py`** - The Streamlit application that serves as the front-end for the chatbot.
2. **`langchain_helper.py`** - The script that defines the logic for creating the vector database, retrieving the QA chain, and processing queries.

### Files Overview

- **`main.py`**: This file initializes the Streamlit app, manages session states, handles user input, and displays responses from the chatbot.
- **`langchain_helper.py`**: This file contains the core logic for creating a vector database from a CSV, defining the question-answering (QA) chain using LangChain, and handling custom query processing with error handling.

## Requirements

To run the project, ensure you have the following Python libraries installed:

- `streamlit`
- `pandas`
- `faiss-cpu` (or `faiss-gpu` if you're using a GPU)
- `langchain`
- `transformers`
- `torch`

You can install the required dependencies using the following:

```bash
pip install streamlit pandas faiss-cpu langchain transformers torch
```

### Additional Requirements

- **CSV File (`codebasics_faqs.csv`)**: Ensure you have a CSV file containing two columns: `prompt` (the question) and `response` (the answer). This is used to build the vector database.
- **Pre-trained Model**: The chatbot uses a pre-trained model, specifically `"facebook/opt-350m"`, for generating answers. Make sure you have access to the Hugging Face model hub for downloading this model.

## How to Run

1. **Prepare the CSV File**: Ensure you have a CSV file called `codebasics_faqs.csv` with the structure:

   ```csv
   prompt,response
   "What is Python?","Python is a high-level programming language..."
   "How do I use Streamlit?","Streamlit is a Python library for building web apps..."
   ```

2. **Run the Streamlit App**: Once your environment is set up, you can run the Streamlit app by executing the following command:

   ```bash
   streamlit run main.py
   ```

3. **Interacting with the Chatbot**: Open the URL provided by Streamlit (usually `http://localhost:8501/`), type your question in the chat input box, and get the response from the chatbot.

4. **CSV Encoding Issues**: The `create_vector_db` function in `langchain_helper.py` tries multiple encodings (`utf-8`, `latin-1`, `cp1252`, `iso-8859-1`) to read the CSV. If there are issues reading your CSV, ensure it uses one of these common encodings or manually edit the script to support others.

## Files and Functions

### `main.py`

- **`initialize_session_state()`**: Initializes the Streamlit session state, including messages and the QA chain.
- **`main()`**: The main function that sets up the Streamlit app, handles user interaction, and displays the chatbot responses.

### `langchain_helper.py`

- **`create_vector_db()`**: Reads the CSV file (`codebasics_faqs.csv`) and creates a vector database using FAISS. It tries multiple encodings to ensure the CSV is correctly loaded.
- **`get_qa_chain()`**: Initializes the LangChain-based question-answering chain with a pre-trained transformer model for text generation and a vector database for document retrieval.
- **`custom_qa_process()`**: Handles the processing of user queries, searching the vector database for relevant documents, and generating a response based on the context.

## Troubleshooting

- **CSV Load Issues**: If there are issues loading your CSV, check that the file is correctly formatted with `prompt` and `response` columns. You can also modify the encodings tried in the `create_vector_db` function.
- **Model Loading Errors**: If the model fails to load, ensure that you're connected to the internet to download the model from the Hugging Face hub.
- **Performance**: If the chatbot response times are slow, consider using a larger machine or a GPU to speed up the text generation process.

## License

This project is open-source and available under the MIT License. 
