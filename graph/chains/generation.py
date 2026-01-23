from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# 1. Promptu Manuel Tanımla (rlm/rag-prompt muadili)
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 2. LLM ve Chain Yapısı
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
generation_chain = prompt | llm | StrOutputParser()
