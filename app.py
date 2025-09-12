# Create streamlit app file
app_code = '''
import streamlit as st
from pathlib import Path
import sys
sys.path.append('.')

# Import your modules
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

st.title("🤖 Enterprise Q&A System")

# Load vector store
@st.cache_resource
def load_qa_system():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "data/vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

if Path("data/vectorstore/faiss_index").exists():
    qa_chain = load_qa_system()
    
    question = st.text_input("Ask a question:")
    
    if st.button("Submit") and question:
        with st.spinner("Thinking..."):
            result = qa_chain({"query": question})
            
        st.write("### Answer:")
        st.write(result["result"])
        
        st.write("### Sources:")
        for doc in result["source_documents"]:
            source = Path(doc.metadata.get("source", "")).name
            st.write(f"- {source}")
else:
    st.error("Vector store not found. Run the notebook first.")
'''
