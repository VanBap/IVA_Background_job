from chatbot_webdata_only_text import State, retrieve, generate
from langgraph.graph import START, StateGraph

# Compile application and test (CONTROL FLOW)
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "Hướng dẫn tôi xử lý đơn hàng"})
# print(f"response: {response}")
# print(f"Context: {response['context']}")
print(f"answer: {response['answer']}")