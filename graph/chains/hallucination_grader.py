from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from typing import Literal

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generated answer.
    """

    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'",
    )

structered_llm_grader = llm.with_structured_output(GradeHallucinations)

system_prompt = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(

    [
        ("system", system_prompt),
        ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

#hallucination_grader: RunnableSequence = ... Bu type hint (tip belirtme).
hallucination_grader: RunnableSequence =  hallucination_prompt | structered_llm_grader