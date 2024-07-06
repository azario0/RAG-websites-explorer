from flask import Flask, render_template, request, jsonify, session
import requests
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    api_key = request.form['api_key']
    session['api_key'] = api_key
    session['chat_history'] = []
    return jsonify({"status": "success", "message": "API key set successfully"})

@app.route('/process_url', methods=['POST'])
def process_url():
    if 'api_key' not in session:
        return jsonify({"status": "error", "message": "API key not set"})

    url = request.form['url']
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        with open('downloaded_page.html', 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        document = UnstructuredHTMLLoader('downloaded_page.html').load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(document)
        
        embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=session['api_key'])
        
        db = FAISS.from_texts(["", ""], embeddings)
        retv = db.as_retriever()
        
        batch_size = 96
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            retv.add_documents(batch)
        
        db.save_local("faiss_index")
        
        return jsonify({"status": "success", "message": "URL processed successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to download the page"})

@app.route('/ask', methods=['POST'])
def ask():
    if 'api_key' not in session:
        return jsonify({"status": "error", "message": "API key not set"})

    question = request.form['question']
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=session['api_key'])
    embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=session['api_key'])
    
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    
    template = """You are a helpful assistant. Use the following pieces of context from this html page and conversation history to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Document context: {context}
    Conversation history: {chat_history}
    Question: {question}
    Helpful Answer:"""
    
    prompt_template = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retv,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        chain_type="stuff"
    )
    
    result = qa_chain({"question": question, "chat_history": session['chat_history']})
    session['chat_history'].append((question, result["answer"]))
    return jsonify({"answer": result["answer"]})

if __name__ == '__main__':
    app.run(debug=True)