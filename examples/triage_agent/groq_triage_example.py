from microagent.core import Microagent
from microagent.types import Agent
from microagent.repl import run_demo_loop

# Initialize Microagent with Groq as the LLM provider
client = Microagent(llm_type='groq')

# Start of new agent section
def process_refund(item_id, reason="NOT SPECIFIED"):
    """Refund an item. Refund an item. Make sure you have the item_id of the form item_... Ask for user confirmation before processing the refund."""
    print(f"[mock] Refunding item {item_id} because {reason}...")
    return "Success!"

def apply_discount():
    """Apply a discount to the user's cart."""
    print("[mock] Applying discount...")
    return "Applied discount of 11%"

triage_agent = Agent(
    model="llama-3.1-70b-versatile",
    tool_choice="auto",
    name="Triage Agent",
    instructions="Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.",
)
sales_agent = Agent(
    model="llama-3.1-70b-versatile",
    tool_choice="auto",
    name="Sales Agent",
    instructions="Be super enthusiastic about selling bees.",
)
refunds_agent = Agent(
    model="llama-3.1-70b-versatile",
    tool_choice="auto",
    name="Refunds Agent",
    instructions="Help the user with a refund. If the reason is that it was too expensive, offer the user a refund code. If they insist, then process the refund.",
    functions=[process_refund, apply_discount],
)

def transfer_back_to_triage():
    """Call this function if a user is asking about a topic that is not handled by the current agent."""
    return triage_agent

def transfer_to_sales():
    return sales_agent

def transfer_to_refunds():
    return refunds_agent

triage_agent.functions = [transfer_to_sales, transfer_to_refunds]
sales_agent.functions.append(transfer_back_to_triage)
refunds_agent.functions.append(transfer_back_to_triage)

run_demo_loop(triage_agent, llm_type='groq', debug=True)