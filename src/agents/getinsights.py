import asyncio, os

from mcp.client.session_group import StreamableHttpParameters
from oci.addons.adk import Agent, AgentClient
from oci.addons.adk.mcp import MCPClientStreamableHttp
from pathlib import Path
from dotenv import load_dotenv
from oci.addons.adk.tool.prebuilt import AgenticRagTool
from mcp.client.stdio import StdioServerParameters
from oci.addons.adk.mcp import MCPClientStdio
from anyio import get_cancelled_exc_class
from src.prompt_engineering.topics.ask_data import prompt_Agent_Auditor

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent

load_dotenv(PROJECT_ROOT / "config/.env")  # expects OCI_ vars in .env

# Set up the OCI GenAI Agents endpoint configuration
OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE")
OCI_PROFILE = os.getenv("OCI_PROFILE")
AGENT_EP_ID = os.getenv("AGENT_EP_ID")
AGENT_REGION = os.getenv("AGENT_REGION")


async def agent_flow(input_prompt: str):
    # MCP endpoint configs
    redis_server_params = StreamableHttpParameters(
        url="https://d7e385f08598.ngrok-free.app/mcp",
    )

    # time_server_params = StdioServerParameters(
    #     command="uvx",
    #     args=["mcp-server-time", "--local-timezone=America/Los_Angeles"],
    # )

    # Start both MCP clients manually
    # time_mcp_client = MCPClientStdio(params=time_server_params)
    redis_mcp_client = MCPClientStreamableHttp(params=redis_server_params)

    # await time_mcp_client.__aenter__()
    await redis_mcp_client.__aenter__()

    try:
        # Setup agent client
        client = AgentClient(
            auth_type="api_key",
            config=OCI_CONFIG_FILE,
            profile=OCI_PROFILE,
            region=AGENT_REGION
        )

        agent = Agent(
            client=client,
            agent_endpoint_id=AGENT_EP_ID,
            instructions=prompt_Agent_Auditor,
            tools=[
                # await time_mcp_client.as_toolkit(),
                await redis_mcp_client.as_toolkit()
            ],
        )

        agent.setup()



        print(f"Running: {input_prompt}")
        try:
            response = await agent.run_async(input_prompt)
            response.pretty_print()
        except get_cancelled_exc_class():
            print("🟡 Agent run cancelled (tool timeout or interrupt).")

    finally:
        # Clean shutdown with full cancel error suppression
        # try:
        #     await time_mcp_client.__aexit__(None, None, None)
        # except get_cancelled_exc_class():
        #     print("⚠️ MCPClientStdio cancelled during shutdown.")

        try:
            await redis_mcp_client.__aexit__(None, None, None)
        except get_cancelled_exc_class():
            print("⚠️ MCPClientStreamableHttp cancelled during shutdown.")

    return response

if __name__ == "__main__":
    # Test input
    input_message = (
        # "What is the local time now? Which invoice should I pay first "
        "based on criteria such as highest amount due and highest past due date for 'session:e5f6a932-6123-4a04-98e9-6b829904d27f'"
    )
    asyncio.run(agent_flow(input_message))  # ✅ safe now with proper async context use