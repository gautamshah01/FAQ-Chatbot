import streamlit as st
import os
import sys
import traceback

# Append the current directory to sys.path to ensure local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the existing chatbot logic
from langchain_helper  import create_vector_db, get_qa_chain, custom_qa_process


def initialize_session_state():
    """Ensure all required keys exist in session state."""
    # Initialize messages if not already present
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Initialize QA chain if not already present
    if 'qa_chain' not in st.session_state:
        try:
            # Check if vector database exists, if not create it
            if not os.path.exists("faiss_index"):
                with st.spinner('Initializing knowledge base...'):
                    create_vector_db()

            # Create and store QA chain
            st.session_state.qa_chain = get_qa_chain()
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            st.error(traceback.format_exc())
            st.session_state.qa_chain = None


def main():
    # Set page configuration
    st.set_page_config(
        page_title="EdTech FAQ Chatbot",
        page_icon="💬",
        layout="centered"
    )

    # Initialize session state
    initialize_session_state()

    # Title and description
    st.title("💬CodeBasics FAQ Chatbot")
    st.write("Ask questions about our EdTech platform and get instant answers!")

    # Check if QA chain is initialized
    if not hasattr(st.session_state, 'qa_chain') or st.session_state.qa_chain is None:
        st.error("Failed to initialize the chatbot. Please restart the application.")
        return

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about our EdTech platform"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query and get response
        with st.spinner('Thinking...'):
            try:
                # Use the stored QA chain
                response = custom_qa_process(st.session_state.qa_chain, prompt)
            except Exception as e:
                response = f"An error occurred: {e}"
                st.error(traceback.format_exc())

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)


# Ensure proper initialization when run as a script
if __name__ == "__main__":
    main()