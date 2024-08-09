import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import docx2txt

# Load the model (you can use other models as well)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
index = None
documents = []

# Function to process and embed documents
def process_document(file):
    global index, documents

    # Read the content from the uploaded file
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)
    elif file.type == "text/plain":
        text = file.read().decode('utf-8')
    else:
        st.error("Unsupported file format.")
        return

    # Split the document into sentences or chunks
    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]

    # Embed each chunk
    chunk_embeddings = model.encode(chunks)
    
    # Update FAISS index
    if index is None:
        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(chunk_embeddings)
    else:
        index.add(chunk_embeddings)

    # Store the chunks with their embeddings
    documents.extend(chunks)

# Streamlit UI
st.title("Streamlit Document Query Application")

# Upload document
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file:
    process_document(uploaded_file)
    st.success("Document processed successfully!")

# Query input
query = st.text_input("Enter your query:")
if query:
    if index is not None:
        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k=5)
        st.write("### Answers:")
        for i in I[0]:
            st.write(documents[i])
    else:
        st.warning("Please upload a document first.")

# Allow users to download their query history
if st.button("Download Chat History"):
    chat_history = "\n".join([f"Query: {query}\nAnswer: {documents[i]}" for i in I[0]])
    st.download_button(
        label="Download Chat History",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain",
    )
