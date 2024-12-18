from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import pandas as pd
import difflib

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb_file_path = "faiss_index"


def create_vector_db():
    # Try multiple encodings
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings_to_try:
        try:
            # Try reading the CSV with the current encoding
            df = pd.read_csv('codebasics_faqs.csv', encoding=encoding)

            # Verify the DataFrame has the expected columns
            if 'prompt' not in df.columns or 'response' not in df.columns:
                print(f"CSV columns: {df.columns}")
                raise ValueError("CSV must have 'prompt' and 'response' columns")

            # Create documents from the DataFrame
            documents = [
                f"Question: {row['prompt']}\nAnswer: {row['response']}"
                for _, row in df.iterrows()
            ]

            # Create metadata for each document
            metadatas = [
                {"source": "FAQ", "prompt": row['prompt'], "response": row['response']}
                for _, row in df.iterrows()
            ]

            # Create vector store
            vectordb = FAISS.from_texts(
                texts=documents,
                embedding=embeddings,
                metadatas=metadatas
            )
            vectordb.save_local(vectordb_file_path)

            print(f"Successfully loaded CSV with {encoding} encoding")
            return

        except (UnicodeDecodeError, ValueError) as e:
            print(f"Failed to read CSV with {encoding} encoding: {e}")

    # If all encoding attempts fail
    raise RuntimeError("Could not read the CSV file with any of the tried encodings")


def get_qa_chain():
    # Load the vector database
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Increase number of retrieved documents
    )

    # Use a more suitable model for QA
    model_name = "facebook/opt-350m"  # A smaller, more focused model

    # Create text generation pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model_name,
        max_length=512,
        do_sample=True,
        temperature=0.7
    )

    # Convert pipeline to LangChain LLM
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt_template = """Use ONLY the context below to answer the question. 
    If the answer is not in the context, respond with "I don't know".

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


def custom_qa_process(chain, query):
    try:
        # Perform the query
        result = chain(query)

        # Check if source documents exist
        if not result['source_documents']:
            return "I don't know"

        # Extract possible answers from source documents
        possible_answers = []
        for doc in result['source_documents']:
            # Try to extract the answer from the document
            parts = doc.page_content.split('Answer: ')
            if len(parts) > 1:
                possible_answers.append(parts[1].strip())

        # If no answers found, return "I don't know"
        if not possible_answers:
            return "I don't know"

        # Similarity check to ensure relevance
        def is_similar_enough(query, document_question, threshold=0.6):
            # Use difflib to calculate similarity
            similarity = difflib.SequenceMatcher(None,
                                                 query.lower(),
                                                 document_question.lower()
                                                 ).ratio()
            return similarity >= threshold

        # Check if any of the source documents' questions are similar to the input query
        similar_docs = [
            doc for doc in result['source_documents']
            if is_similar_enough(query, doc.metadata['prompt'])
        ]

        # If no similar documents found, return "I don't know"
        if not similar_docs:
            return "I don't know"

        # Return the first non-empty answer
        return next((ans for ans in possible_answers if ans.strip() and ans.lower() != "i don't know"), "I don't know")

    except Exception as e:
        print(f"An error occurred: {e}")
        return "I don't know"


if __name__ == "__main__":
    # Recreate vector database each time to ensure fresh data
    create_vector_db()

    # Create QA chain
    qa_chain = get_qa_chain()

    # Interactive loop
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'exit':
            break

        answer = custom_qa_process(qa_chain, query)
        print("Answer:", answer)