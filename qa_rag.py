#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install required packages
get_ipython().system('pip install -q langchain langchain-openai openai tiktoken pypdf faiss-cpu streamlit')

import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown

print("✅ Packages installed")


# ## Set API Key

# In[3]:


# Set your OpenAI API key
import getpass
os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter OpenAI API Key: ')
print("✅ API key set")


# ## Setup Folder Structure

# In[32]:


# Create necessary directories
folders = ['data/documents', 'data/vectorstore', 'logs']
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    
print("📁 Folder structure created:")
print("├── data/")
print("│   ├── documents/   (place PDFs here)")
print("│   └── vectorstore/")
print("└── logs/")


# ## Document Processor

# In[38]:


from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, docs_path='data/documents', chunk_size=500, chunk_overlap=50):
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.chunks = []
    
    def load_documents(self):
        """Load all documents from directory"""
        documents = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.docs_path, 
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        # Load text files
        txt_loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents.extend(txt_loader.load())
        
        self.documents = documents
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"📄 Created {len(self.chunks)} chunks")
        return self.chunks

processor = DocumentProcessor()


# ## Create Sample Documents

# In[42]:


# Create sample documents if none exist
sample_content = """
EMPLOYEE HANDBOOK

Vacation Policy:
- Full-time employees receive 15 days of paid vacation per year
- Vacation days accrue monthly at 1.25 days per month
- Unused vacation can roll over up to 5 days

Work Hours:
- Standard hours are 9 AM to 5 PM, Monday through Friday
- Flexible hours available with manager approval

Remote Work:
- Employees may work remotely up to 2 days per week
- Full remote work requires director approval
"""

# Save sample
sample_path = Path('data/documents/sample_handbook.txt')
sample_path.write_text(sample_content)
print(f"✅ Sample document created at {sample_path}")


# ## Process Documents

# In[44]:


# Load and process documents
docs = processor.load_documents()
chunks = processor.split_documents()

# Display first chunk as example
if chunks:
    display(Markdown("**First chunk preview:**"))
    print(chunks[0].page_content[:200] + "...")


# ## Create Vector Store

# In[46]:


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.store_path = 'data/vectorstore/faiss_index'
    
    def create_vectorstore(self, chunks):
        """Create vector store"""
        print("🔄 Creating embeddings...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print(f"✅ Vector store created with {len(chunks)} chunks")
        return self.vectorstore
    
    def save(self):
        """Save to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(self.store_path)
            print(f"💾 Saved to {self.store_path}")
    
    def load(self):
        """Load from disk"""
        if Path(self.store_path).exists():
            self.vectorstore = FAISS.load_local(
                self.store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded existing vector store")
            return True
        return False

vs_manager = VectorStoreManager()
vectorstore = vs_manager.create_vectorstore(chunks)
vs_manager.save()


#  ## Q&A System

# In[48]:


from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class QASystem:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.setup_chain()
        self.history = []
    
    def setup_chain(self):
        prompt_template = """Use the context to answer the question. 
        If you don't know, say so. Mention source documents when possible.
        
        Context: {context}
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask(self, question):
        """Ask a question"""
        result = self.qa_chain({"query": question})
        
        # Save to history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': result['result']
        })
        
        return result

qa_system = QASystem(vectorstore)
print("✅ Q&A System ready")


# ## Interactive Q&A Function

# In[50]:


def ask_question(question):
    """Interactive Q&A interface"""
    display(Markdown(f"**❓ Question:** {question}"))
    
    result = qa_system.ask(question)
    
    display(Markdown(f"**💡 Answer:** {result['result']}"))
    
    # Show sources
    if result.get('source_documents'):
        display(Markdown("**📚 Sources:**"))
        for i, doc in enumerate(result['source_documents'], 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"{i}. {Path(source).name}")

# Test
ask_question("What is the vacation policy?")


# ## Batch Questions

# In[52]:


# Test multiple questions
test_questions = [
    "What is the vacation policy?",
    "Can I work remotely?",
    "What are the work hours?"
]

for q in test_questions:
    ask_question(q)
    print("-" * 50)



#  ## Save Session History

# In[53]:


# Save Q&A history
if qa_system.history:
    df = pd.DataFrame(qa_system.history)
    df.to_csv('logs/qa_history.csv', index=False)
    display(df)
    print(f"✅ Saved {len(df)} Q&A pairs to logs/qa_history.csv")


# In[ ]:




