from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import os
from fastapi.responses import JSONResponse
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent

# Load .env file
load_dotenv()

from runnable import get_runnable

app = FastAPI(
    title="LangServe AI Agent",
    version="1.0",
    description="LangGraph backend for the AI Agents Masterclass series agent.",
)

# @app.post("/info")
@app.api_route("/info", methods=["GET", "POST"])
async def info():
    return JSONResponse(content={"status": "ok"})

@app.get("/")
async def root():
    return {"status": "FastAPI agent is online!"}


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def main():
    # Fetch the AI Agent LangGraph runnable which generates the workouts
    runnable = get_runnable()

    # Create the Fast API route to invoke the runnable
    add_routes(
        app,
        runnable
    )

    # Fetch the LangGraph graph for CopilotKit
    langgraph_graph = get_runnable()

    # Initialize CopilotKit SDK with the LangGraph agent
    copilotkit_sdk = CopilotKitRemoteEndpoint(
        agents=[
            LangGraphAgent(
                name="ai_agent",
                description="LangGraph agent for CopilotKit integration.",
                graph=langgraph_graph,
            )
        ],
    )

    # Add CopilotKit endpoint to FastAPI
    add_fastapi_endpoint(app, copilotkit_sdk, "/copilotkit")

    # Start the API
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()