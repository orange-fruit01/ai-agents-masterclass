from langchain_core.tools import tool
from typing import List, Dict, Any
from research_canvas.crewai.agent import ResearchCanvasFlow
import asyncio

@tool
def run_research_agent(research_question: str, report: str, messages: List[Dict[str, Any]]) -> dict:
    """
    Tool to run the Research Canvas CrewAI agent flow as a tool.

    Args:
        research_question (str): The research question to investigate.
        report (str): The initial report or context.
        messages (List[Dict[str, Any]]): The chat history/messages.

    Returns:
        dict: The final state after running the agent, including updated messages, report, and resources.
    """
    # Prepare the initial state
    state = {
        "research_question": research_question,
        "report": report,
        "messages": messages,
    }
    # Run the flow synchronously
    flow = ResearchCanvasFlow(state)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(flow.start())
        result = loop.run_until_complete(flow.chat())
        loop.run_until_complete(flow.end())
    finally:
        loop.close()
    return flow.state


available_agent_tools = {
    "run_research_agent": run_research_agent
} 