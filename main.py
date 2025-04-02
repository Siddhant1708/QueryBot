from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
# from tools import search_tool, wiki_tool

load_dotenv()

#We want our response from the LLM in this format
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGroq(model_name="llama-3.3-70b-versatile")
#what this parser do is like it parses the response of the llm in ReasearchResponse object format
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


#this is the format of our prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
# what this partial do is, it takes pydantic_object ResearchResponse and convert it in to string
# and pass this string as format_instructions in the prompt string 


# Now we have our prompt, LLm and Parser
# lets create and agent
# tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)




try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
   
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)

