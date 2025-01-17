from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load PDF and preprocess
loader = PyPDFLoader("lebo105.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create vector store and retriever
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Define system prompt
system_prompt = (
    "You are an assistant for making questions. "
    "Use the following pieces of retrieved context to make questions. "
    "Make multiple-choice questions with 4 options. "
    "Give 5 questions. "
    "Make different questions every time. "
    "\n\n"
    "{context}"
)

# Create prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# API route for querying
@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Generate response
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": user_query})
    return jsonify({"questions": response["answer"]})

# Default route
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the RAG Flask API!"})

if __name__ == "__main__":
    app.run()
