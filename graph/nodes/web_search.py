from typing import Any,Dict
from langchain_core.documents import Document
from graph.state import GraphState
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearchResults(k=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("-----WEB SEARCH-----")

    question = state["question"]
    # Eğer documents anahtarı yoksa boş bir liste döndür, varsa mevcut olanı al
    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query":question})
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_result,metadata={"source": "web", "tool": "tavily"})
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}