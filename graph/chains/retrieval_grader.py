from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    binary_score : str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

#LLM'e "Bana rastgele metin üretme, yukarıda tanımladığım GradeDocuments şemasına uygun bir JSON objesi üret" talimatı verir.
structered_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = ("""You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n 
If the document contains keyword or semantic meaning related to question, grade it as relevant.\n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.""")

grade_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", system_prompt),
    ("human","Retrieved document: {document} User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structered_llm_grader