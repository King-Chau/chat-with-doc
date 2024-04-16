# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 


load_dotenv()

def get_vectorstore_from_url(url):
    file_save = "file.txt"  # the name you want to save file as
    resp = requests.get(url) # making requests to server
    with open(file_save, "wb") as f:   # opening a file handler to create new file
        f.write(resp.content)
    # get the text in document form
    loader = TextLoader(file_save)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"])
    document_chunks = text_splitter.split_documents(document)
    print(document_chunks)
    
    # create a vectorstore from the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(document_chunks, embeddings)

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    prompt = ChatPromptTemplate.from_messages([
      ("user", "{input}"),
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    
    prompt = ChatPromptTemplate.from_template("""仅根据上下文，回答问题:
    <context>
    {context}
    </context>
    问题: {input}""")
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "input": user_query
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="文档问答", page_icon="🤖")
st.title("文档问答")

# sidebar
with st.sidebar:
    st.header("设置")
    website_url = st.text_input("公网可访问的文件URL")

if website_url is None or website_url == "":
    st.info("仅供效果评测，请不要大量调用，谢谢！请在左侧输入文件URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="你好，请输入问题"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    

    # user input
    user_query = st.chat_input("输入关于文档的问题...")
    if user_query is not None and user_query != "":
        with st.spinner('处理中：' + user_query):
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
        
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
