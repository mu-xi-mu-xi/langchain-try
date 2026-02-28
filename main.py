
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_models import ChatZhipuAI

checkpoint=MemorySaver()
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatZhipuAI(
    temperature=0.5,
    api_key="your api key", 
    model="GLM-4-Flash"
)
agent = create_agent(
    model=model,
    tools=[get_weather],
    checkpointer=checkpoint
)
config={"configurable":{"thread_id":"1"}}
# Run the agent
res=agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config=config
)
print(res)

