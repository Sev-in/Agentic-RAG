from dotenv import load_dotenv
load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    state= app.invoke(input={"question":"what is pizza?"})
    print(state["generation"])

