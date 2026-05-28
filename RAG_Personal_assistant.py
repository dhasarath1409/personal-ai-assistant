from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_react_agent,AgentExecutor
from ddgs import DDGS    
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatOpenAI(
    model ="openrouter/free",
    api_key =os.getenv("OPENROUTER_API_KEY"),
    base_url = "https://openrouter.ai/api/v1",
    temperature= 0.7
)
print("Loading the Knowledge base...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "knowledge_info.txt")

loader=TextLoader(KNOWLEDGE_PATH)
documents=loader.load()

splitter = CharacterTextSplitter(separator="\n",chunk_size=200,chunk_overlap=20)
chunks= splitter.split_documents(documents)

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_stores=FAISS.from_documents(chunks,embeddings)

retriever=vector_stores.as_retriever()

def ddg_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([
                    f"{r['title']}: {r['body']}" 
                    for r in results
                ])
            return "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"

def calculate (expression:str)->str:
    try:
        expression = expression.strip().strip('"').strip("'")
        result=eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

search_tool=Tool(name="websearch",func=ddg_search,description="Search the internet for current or real-time information. Use for news, recent events, or anything that may have changed recently.")
calc_tool=Tool(name="calculator",func=calculate,description="Evaluate math expressions. Input must be a valid math expression like '25 * 48' or '(100 + 50) / 3'.")

tools=[search_tool,calc_tool]

react_prompt=PromptTemplate.from_template("""You are Aiden, a helpful Personal AI Assistant.
Answer the question as best you can using the tools available.

You have access to the following tools:
{tools}

Use this format:

Question: the input question
Thought: think about what to do
Action: the tool to use, one of [{tool_names}]
Action Input: the input to the tool
Observation: the result from the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

agent=create_react_agent(llm,tools,react_prompt)

agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=7
)

system_prompt=SystemMessage(content="""
You are Aiden, a helpful and friendly Personal AI Assistant.
You help the user with questions, research, math, and knowledge lookups.
You are concise, clear, and always honest.
If you don't know something, you say so.
"""
)

chat_history=[system_prompt]

print("Aiden your Personal Chat Assistant")
print("Type EXIT to quit")

if __name__=="__main__":
    print("Starting Aiden in terminal mode...")

    while True:
        user_input=input("You : ").strip()

        if user_input.lower()=="exit":
            print("Aiden: Goodbye! See you later")
            break
        if not user_input:
            continue
    
        retrieved_docs=retriever.invoke(user_input)

        context="\n".join([doc.page_content for doc in retrieved_docs])

        history_str=""
        for msg in chat_history[1:]:
            if isinstance(msg,HumanMessage):
                history_str+=f"Human:{msg.content}\n"
            elif isinstance(msg,AIMessage):
                history_str+=f"Assistant:{msg.content}\n"

        full_input=f"""
    Previous conversation:{history_str}
    knowledge context(use if relevant, ignore if not):{context}
    current Question:{user_input}"""

        chat_history.append(HumanMessage(content=user_input))

        try:
            response=agent_executor.invoke({"input":full_input})
            reply=response["output"]

            if "Agent stopped" in reply:
                raise ValueError("Agent hit itteration limit - fallback")

        except Exception as e:
            print(f"\n Agent fallback activated")

            fallback_input=f"""
    Use the context if it is relevant, ignore if it not:{context}
    Question:{user_input}"""

            fallback_message=chat_history+[HumanMessage(content=fallback_input)]
            fallback_response=llm.invoke(fallback_message)
            reply=fallback_response.content

        chat_history.append(AIMessage(content=reply))

        print(f"\nAiden : {reply}\n")


