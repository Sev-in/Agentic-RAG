from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq

class RouterQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    datasource: Literal["vectorstore","websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

structered_llm_router = llm.with_structured_output(RouterQuery)

system = """
You are an expert at routing a user question to a vectorstore or web search.\n 
The vectorstore contains documents related to agents, prompt engineering and adversial attacks.\n 
Use the vectorstore for questions on these topics. For all else, use web-search.\n 
"""

route_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","{question}"),
    ]
)

question_router = route_prompt | structered_llm_router