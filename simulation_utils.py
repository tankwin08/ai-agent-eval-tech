"""
Utility functions for simulation notebook:
- Converts LangChain message format to OpenAI message format
- Provides simulated user creation and chat simulator creation helpers
"""

from typing import List, Dict, Tuple, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
# from langgraph import LangGraphSimulator


def langchain_to_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Converts a list of LangChain-style messages to OpenAI Chat API format.

    Args:
        messages: List of dicts with keys 'role' and 'content'

    Returns:
        List of dicts in OpenAI message format
    """
    role_map = {
        "user": "user",
        "assistant": "assistant",
        "system": "system"
    }
    openai_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        openai_msgs.append({"role": role_map.get(role, "user"), "content": content})
    return openai_msgs


def create_simulated_user(system_prompt_template: str, llm) -> RunnableLambda:
    """
    Creates a simulated user agent using the provided LLM and prompt template.

    Args:
        system_prompt_template: String template for the system prompt, with {instructions} placeholder
        llm: The LangChain ChatOpenAI or compatible LLM instance

    Returns:
        A RunnableLambda that takes {'instructions': ..., 'messages': ...} and returns AIMessage
    """

    def _invoke(args: Dict[str, Any]) -> AIMessage:
        instructions = args["instructions"]
        messages = args["messages"]
        # Convert messages to LangChain messages for LLM
        lc_messages = []
        for role, content in messages:
            if role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
        # Prepare prompt
        system_prompt = system_prompt_template.format(instructions=instructions)
        prompt_msgs = [SystemMessage(content=system_prompt)]
        prompt_msgs.extend(lc_messages)
        # Call LLM
        response = llm.invoke(prompt_msgs)
        return response if isinstance(response, AIMessage) else AIMessage(content=response.content)

    return RunnableLambda(_invoke)


def create_chat_simulator1(assistant_fn, simulated_user, input_key="input", max_turns=10):
    """
    Creates a simulation harness that alternates between assistant and simulated user.

    Args:
        assistant_fn: Function taking list of messages and returning string (assistant response)
        simulated_user: RunnableLambda for the simulated user
        input_key: Key to use for first user message
        max_turns: Maximum allowed conversation turns

    Returns:
        A generator function that streams conversation events.
    """
    def _stream_simulation(example: Dict[str, Any]):
        instructions = example.get("instructions")
        input_msg = example.get(input_key, "")
        messages = [{"role": "user", "content": input_msg}]
        turns = 0
        finished = False
        while turns < max_turns and not finished:
            # Assistant response
            assistant_msg = assistant_fn(messages)
            messages.append({"role": "assistant", "content": assistant_msg})
            yield {"assistant": {"messages": messages.copy()}}
            # Simulated user response
            # Convert messages for simulated_user (tuple format)
            tuple_msgs = []
            for msg in messages:
                tuple_msgs.append((msg["role"], msg["content"]))
            user_response = simulated_user.invoke({
                "instructions": instructions,
                "messages": tuple_msgs
            })
            user_content = user_response.content if hasattr(user_response, "content") else str(user_response)
            messages.append({"role": "user", "content": user_content})
            yield {"user": {"messages": messages.copy()}}
            if user_content.strip().upper() == "FINISHED":
                finished = True
            turns += 1
        yield {"__end__": {"messages": messages.copy()}}

    class Simulator:
        def stream(self, example):
            return _stream_simulation(example)

    return Simulator()


# If you're using LangGraph or your own simulator class, import it here.


def _create_langgraph_simulator(assistant, simulated_user, input_key="input", max_turns=10):
    """
    Stub for LangGraph simulator construction.
    Replace this with your actual LangGraph graph building logic.
    Should return an object with a .stream(input_dict) method.
    """
    class DummySimulator:
        def stream(self, input_dict):
            # Dummy conversation: replace with your actual simulation loop
            messages = []
            # Simulate a turn-based conversation for demonstration
            for turn in range(min(max_turns, 2)):
                messages.append({"role": "user", "content": f"user says: {input_dict[input_key]}"})
                messages.append({"role": "assistant", "content": f"assistant replies: {input_dict[input_key]}"})
            messages.append({"role": "user", "content": "FINISHED"})
            # Emulate LangGraph streaming event format
            for m in messages:
                yield {"assistant" if m["role"] == "assistant" else "user": {"messages": [m]}}
            yield {"__end__": {}}
    return DummySimulator()

def create_chat_simulator(assistant, simulated_user, input_key="input", max_turns=10):
    """
    Returns a factory function suitable for LangSmith's run_on_dataset.
    The factory takes an input dict and returns a dict with 'messages'.
    """
    simulator = _create_langgraph_simulator(assistant, simulated_user, input_key=input_key, max_turns=max_turns)
    
    def simulator_factory(example: dict) -> dict:
        """
        example: dict with keys like 'input', 'instructions'.
        Returns: dict with 'messages' (conversation transcript).
        """
        events = simulator.stream({
            "input": example[input_key],
            "instructions": example["instructions"],
        })
        messages = []
        for event in events:
            if "__end__" in event:
                break
            # Each event is a dict: {role: {"messages": [message_dict]}}
            role, state = next(iter(event.items()))
            next_message = state["messages"][-1]
            messages.append(next_message)
        return {"messages": messages}
    
    return simulator_factory