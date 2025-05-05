'''
import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Path to FAISS database
DB_FAISS_PATH = "chatbot/vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """
    Load and return the vectorstore for document retrieval
    """
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(conversation_history, question):
    """
    Format the custom prompt with conversation history and current question.
    """
    # Combine the conversation history into a single string (limiting the size if necessary)
    context = "\n".join(conversation_history[-5:])  # Limiting the history to the last 5 messages
    CUSTOM_PROMPT_TEMPLATE = """
    Use the following context to answer the question as accurately as possible.
    Be concise but complete. If the answer is not in the context, respond with "I don't know". 
    Don't make up information. Only use what is relevant from the context.

    Context: {context}
    Question: {question}

    Answer:
    """

    # Create the prompt with the updated context and question
    prompt = PromptTemplate(
        input_variables=["context", "question"],  # Define the input variables
        template=CUSTOM_PROMPT_TEMPLATE
    )
    # Return the formatted prompt object (not the string itself)
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    """
    Load the Hugging Face model using the given repo_id and token.
    """
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

def main():
    """
    Main function to run the chatbot interface.
    """
    st.title("🩺 Medical Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Take user input
    prompt = st.chat_input("Type your medical query here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            with st.spinner("Thinking... 🤖"):
                # Load vectorstore for retrieving relevant documents
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                # Include the entire conversation history in the prompt
                conversation_history = [message['content'] for message in st.session_state.messages]

                # Set custom prompt using conversation history
                prompt_template = set_custom_prompt(conversation_history, prompt)

                # Setup the retrieval-based question-answering chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': prompt_template}  # Passing PromptTemplate object
                )

                # Get the response from the chain
                response = qa_chain.invoke({'query': prompt})

                result = response["result"]
                print("🔍 RAW RESPONSE:", result)  # For debugging in console/logs

                # Display the result to the user
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()






import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# URLs for deployed tools
tablet_info_url = "https://5df9nqf5y2dqv2w7ev6xza.streamlit.app/"
lab_report_url = "https://fxvmph62r948bafhhj8meh.streamlit.app/"

# Path to FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """
    Load and return the vectorstore for document retrieval
    """
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def detect_tool_intent(prompt):
    """
    Detects if the user's query is specifically about a medicine or a lab report.
    Returns 'tablet', 'lab', or None
    """
    lower_prompt = prompt.lower()

    info_intent_keywords = ["can you tell me about this", "can you give me info", "can you give me some information", "can you give me some details",  "can i know about my", "learn about"]

    tablet_keywords = ["tablet", "medicine", "drug", "pill", "capsule", "medication", "dose"]
    lab_keywords = ["report", "test", "lab", "cbc", "blood", "x-ray", "scan", "result", "ecg", "mri", "radiology", "urine", "thyroid"]

    if any(word in lower_prompt for word in tablet_keywords):
        return "tablet"
    elif any(word in lower_prompt for word in lab_keywords):
        return "lab"

    return None

def set_custom_prompt(conversation_history, question):
    """
    Format the custom prompt with conversation history and current question.
    """
    context = "\n".join(conversation_history[-5:])
    CUSTOM_PROMPT_TEMPLATE = """
    Use the following context to answer the question as accurately as possible.
    Be concise but complete. If the answer is not in the context, respond with "I don't know". 
    Don't make up information. Only use what is relevant from the context.

    Context: {context}
    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=CUSTOM_PROMPT_TEMPLATE
    )
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    """
    Load the Hugging Face model using the given repo_id and token.
    """
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

def main():
    """
    Main function to run the chatbot interface.
    """
    st.title("🩺 Medical Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your medical query here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Check if intent matches tablet or lab-related queries
        intent = detect_tool_intent(prompt)
        if intent == "tablet":
            tablet_response = f"It looks like you're asking about a medication. Please visit our [Tablet Info Summarizer]({tablet_info_url}) for detailed insights. 💊"
            st.chat_message('assistant').markdown(tablet_response)
            st.session_state.messages.append({'role': 'assistant', 'content': tablet_response})
            return
        elif intent == "lab":
            lab_response = f"You can upload your report to our [Lab Report Analyzer]({lab_report_url}) for detailed interpretation. 📊"
            st.chat_message('assistant').markdown(lab_response)
            st.session_state.messages.append({'role': 'assistant', 'content': lab_response})
            return

        # LLM + Vector QA Flow
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            with st.spinner("Thinking... 🤖"):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                conversation_history = [message['content'] for message in st.session_state.messages]
                prompt_template = set_custom_prompt(conversation_history, prompt)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': prompt_template}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                print("🔍 RAW RESPONSE:", result)

                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
'''




import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# URLs for deployed tools
tablet_info_url = "https://5df9nqf5y2dqv2w7ev6xza.streamlit.app/"
lab_report_url = "https://fxvmph62r948bafhhj8meh.streamlit.app/"
diet_info_url = "https://8z3sw876xnebjntbj26r8b.streamlit.app/"

# Path to FAISS database
DB_FAISS_PATH = "chatbot/vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def detect_tool_intent(prompt):
    lower_prompt = prompt.lower()

    tablet_keywords = ["tablet"]
    lab_keywords = ["report", "test", "lab", "cbc", "blood", "x-ray", "scan", "result", "ecg", "mri", "radiology", "urine", "thyroid"]
    diet_keywords = ["diet"]

    if any(word in lower_prompt for word in tablet_keywords):
        return "tablet"
    elif any(word in lower_prompt for word in lab_keywords):
        return "lab"
    elif any(word in lower_prompt for word in diet_keywords):
        return "diet"

    return None

def set_custom_prompt(conversation_history, question):
    context = "\n".join(conversation_history[-5:])
    CUSTOM_PROMPT_TEMPLATE = """
    Use the following context to answer the question as accurately as possible.
    Be concise but complete. If the answer is not in the context, respond with "I don't know". 
    Don't make up information. Only use what is relevant from the context.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=CUSTOM_PROMPT_TEMPLATE
    )
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

def main():
    st.title("🩺 Curabot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your medical query here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # 🔸 Check for casual responses like "ok", "thanks", etc.
        casual_responses = ["ok", "okay", "okkk", "cool", "thanks", "thank you", "great", "got it", "hmm", "alright"]
        if prompt.lower().strip() in casual_responses:
            st.chat_message('assistant').markdown("👍")
            st.session_state.messages.append({'role': 'assistant', 'content': "👍"})
            return

        # 🔸 Check if intent matches tablet or lab-related queries
        intent = detect_tool_intent(prompt)
        if intent == "tablet":
            tablet_response = f"It looks like you're asking about a medication. Please visit our [Tablet Info Summarizer]({tablet_info_url}) for detailed insights. 💊"
            st.chat_message('assistant').markdown(tablet_response)
            st.session_state.messages.append({'role': 'assistant', 'content': tablet_response})
            return
        elif intent == "lab":
            lab_response = f"You can upload your report to our [Lab Report Analyzer]({lab_report_url}) for detailed interpretation. 📊"
            st.chat_message('assistant').markdown(lab_response)
            st.session_state.messages.append({'role': 'assistant', 'content': lab_response})
            return
        elif intent == "diet":
            diet_response = f"Do you Want to know more about diet...Click on this and Explore. [Diet Recommendation]({diet_info_url}) for detailed information. 🍴"
            st.chat_message('assistant').markdown(diet_response)
            st.session_state.messages.append({'role': 'assistant', 'content': diet_response})
            return

        # 🔸 Vectorstore + LLM Flow
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            with st.spinner("Thinking... 🤖"):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                conversation_history = [message['content'] for message in st.session_state.messages]
                prompt_template = set_custom_prompt(conversation_history, prompt)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': prompt_template}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                print("🔍 RAW RESPONSE:", result)

                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
