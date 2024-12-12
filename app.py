from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini API key
genai.configure(api_key="AIzaSyBGj1rmJqOmahwj_2b5upsfln2qg5T_SOA")

# Function to load and extract text from PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Function to split text into smaller chunks for better processing
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Load your knowledge base PDF
extracted_data = load_pdf("C:\\Users\\aksha\\OneDrive\\Desktop\\sih_cb\\Q and A.pdf")

# Split the extracted text into smaller chunks
text_chunks = text_split(extracted_data)

# Function to create context from the knowledge base
def create_context_from_knowledge_base(query, text_chunks):
    # Create context from the first few chunks. This can be adjusted based on your need.
    context = "\n".join([t.page_content for t in text_chunks[:3]])  # Adjust this logic for larger context if needed
    return context

# Function to query the Gemini model
def ask_gemini(query, context):
    # Generative model setup
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Create the prompt by combining context and question in a conversational format
    prompt = f"""
    Context: {context}
    
    Question: {query}
    
    Please provide a helpful and concise answer based on your knowledge, avoiding references to specific text.
    """

    # Generate the response from Gemini
    response = model.generate_content(prompt)
    return response.text

# Route for the homepage to display the chatbot UI
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle user queries and generate chatbot responses
@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form['query']
    context = create_context_from_knowledge_base(user_query, text_chunks)
    response = ask_gemini(user_query, context)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
