import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from RAG_Personal_assistant import agent_executor, retriever, llm, chat_history

st.set_page_config(
    page_title="Aiden – Personal AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Aiden — Personal AI Assistant")
st.caption("Powered by Llama 3.3 70B · RAG · DuckDuckGo · Calculator")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask Aiden anything...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    history_str = ""
    for msg in chat_history[1:]:
        if isinstance(msg, HumanMessage):
            history_str += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Assistant: {msg.content}\n"

    full_input = f"""
Previous conversation: {history_str}
Knowledge context (use if relevant, ignore if not): {context}
Current Question: {user_input}"""

    chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Aiden is thinking..."):

            try:
                response = agent_executor.invoke({"input": full_input})
                reply = response["output"]

                if "Agent stopped" in reply:
                    raise ValueError("Agent hit iteration limit — fallback")

            except Exception as e:
                fallback_input = f"""
Use the context if it is relevant, ignore if not: {context}
Question: {user_input}"""

                fallback_message = chat_history + [HumanMessage(content=fallback_input)]
                fallback_response = llm.invoke(fallback_message)
                reply = fallback_response.content

        st.markdown(reply)

    chat_history.append(AIMessage(content=reply))
    st.session_state.messages.append({"role": "assistant", "content": reply})