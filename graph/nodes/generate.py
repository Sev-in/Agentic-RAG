from graph.chains.generation import generation_chain
from graph.state import GraphState
from typing import Any, Dict

def generate(state: GraphState) -> Dict[str, Any]:
    print("----GENERATE----")
    question = state["question"]
    documents = state["documents"]
    # SADECE metin içeriklerini birleştir, metadata'dan kurtul!
    docs_txt = "\n\n".join([d.page_content for d in documents])

    # Şimdi temizlenmiş metni gönder
    generation = generation_chain.invoke({"context": docs_txt, "question": question})

    return {"question":question,"documents":documents, "generation":generation}