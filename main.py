from ollama import chat
from ddgs import DDGS
from datetime import datetime
import json
import os


MEMORY_FILE = "memory.json"
MODEL_NAME = "qwen3:4b"


# ============================================================
# CURRENT DATE AND TIME
# ============================================================

now = datetime.now().astimezone()

current_date = now.strftime("%B %d, %Y")
current_time = now.strftime("%H:%M:%S %Z")


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are my personal AI assistant.

Your name is Bluemies.

Be helpful, accurate, and honest.
Keep your answers clear and concise.

Current date: {current_date}
Current local time: {current_time}

Important date instruction:
The current date above comes from the computer running this program.
Do not assume that the current year is the year from your training data.

When web search results are provided:
- Treat the web search results as newer than your training knowledge.
- Use the web results to answer questions about current or recent information.
- Do not replace newer web information with older information from your training memory.
- Pay attention to dates in the search results.
- If the results conflict with your previous knowledge, prefer newer reliable web sources.
- Do not invent information that is not supported by the search results.

Do not claim that you searched the web unless search results were actually provided.
"""


# ============================================================
# MEMORY
# ============================================================

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            messages,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# INTELLIGENT TOOL ROUTER
# ============================================================

def route_request(text):
    router_prompt = f"""
You are the routing system for a local AI assistant.

Your job is to decide what type of action is appropriate for the
user's request.

Return ONLY ONE of these labels:

CHAT
WEB
CALCULATE
CODE
FILE

Rules:

CHAT
Use CHAT for:
- General questions
- Explanations
- Casual conversation
- Writing
- Translation
- Brainstorming
- General knowledge that does not require current information

WEB
Use WEB when the question requires:
- Current information
- Recent information
- News
- Today's information
- Current prices
- Current weather
- Current sports information
- Current political information
- Current products or technology
- Current information about people, companies, games, events, etc.
- Information that may have changed since your training data

CALCULATE
Use CALCULATE when the user wants:
- Arithmetic
- Mathematical calculations
- Unit conversions
- Numerical calculations

CODE
Use CODE when the user wants:
- Code written
- Code explained
- Code debugged
- Programming help
- A script
- A programming solution

FILE
Use FILE when the user wants to:
- Read a file
- Analyze a file
- Search a file
- Modify a file
- Work with a document or local file

Important:
Return ONLY the label.
Do not explain your decision.

User request:
{text}

Label:
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": router_prompt
            }
        ]
    )

    decision = response.message.content.strip().upper()

    # Remove accidental punctuation or extra text
    decision = decision.replace(".", "").replace(":", "").strip()

    valid_routes = {
        "CHAT",
        "WEB",
        "CALCULATE",
        "CODE",
        "FILE"
    }

    if decision in valid_routes:
        return decision

    # If Qwen returns something unexpected,
    # safely fall back to normal conversation.
    print(f"[Router] Unexpected decision: {decision}")
    print("[Router] Falling back to CHAT.")

    return "CHAT"


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query):
    print("\n[Tool] Searching the web...")

    try:
        results = DDGS().text(
            query,
            max_results=5
        )

    except Exception as error:
        print(f"[Tool] Search failed: {error}")
        return "Web search failed."

    if not results:
        return "No search results found."

    formatted_results = []

    for i, result in enumerate(results, start=1):

        formatted_results.append(
            f"""
Result {i}
Title: {result["title"]}
URL: {result["href"]}
Description: {result["body"]}
"""
        )

    return "\n".join(formatted_results)


# ============================================================
# START MEMORY
# ============================================================

messages = load_memory()


# ============================================================
# PROGRAM START
# ============================================================

print("Bluemies - Local AI")
print("Intelligent tool routing is enabled.")
print("Type 'exit' to quit.")
print()


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    user_input = input("You: ")


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if user_input.lower() == "exit":

        save_memory(messages)

        print("Conversation saved.")
        print("Goodbye!")

        break


    # --------------------------------------------------------
    # ROUTE THE REQUEST
    # --------------------------------------------------------

    route = route_request(user_input)

    print(f"[Router] Selected route: {route}")


    # ========================================================
    # WEB ROUTE
    # ========================================================

    if route == "WEB":

        print("[Router] This question requires current information.")

        search_results = web_search(user_input)


        # Add original user question
        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        # Add web results
        messages.append(
            {
                "role": "user",
                "content": f"""
The following information was retrieved from the web just now.

IMPORTANT:
- These search results are newer than your training knowledge.
- Use these results as the primary source for current information.
- Do not replace current web information with older information
  from your training data.
- Pay attention to the dates in the search results.
- If the results conflict with your previous knowledge,
  prefer the newer reliable web source.
- Do not invent information that is not supported by the results.

Web search results:

{search_results}

Now answer the user's original question using the search results.
"""
            }
        )


    # ========================================================
    # CALCULATE ROUTE
    # ========================================================

    elif route == "CALCULATE":

        print("[Router] Calculation detected.")
        print("[Tool] Calculator is not implemented yet.")

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        messages.append(
            {
                "role": "user",
                "content": """
The user requested a calculation.

A dedicated calculator tool is not implemented yet.
Answer the calculation as accurately as possible using your
normal reasoning.

Do not claim that a calculator tool was used.
"""
            }
        )


    # ========================================================
    # CODE ROUTE
    # ========================================================

    elif route == "CODE":

        print("[Router] Programming request detected.")

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        messages.append(
            {
                "role": "user",
                "content": """
The user is asking for programming or coding help.

Provide a clear and useful programming answer.
Include code when appropriate.

Do not claim that an external code execution tool was used.
"""
            }
        )


    # ========================================================
    # FILE ROUTE
    # ========================================================

    elif route == "FILE":

        print("[Router] File operation detected.")
        print("[Tool] File tools are not implemented yet.")

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        messages.append(
            {
                "role": "user",
                "content": """
The user requested a file-related operation.

A dedicated file tool is not implemented yet.
Explain what information or file would be needed to complete
the request.

Do not claim that you accessed a file if you did not.
"""
            }
        )


    # ========================================================
    # CHAT ROUTE
    # ========================================================

    else:

        print("[Router] Normal conversation.")

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


    # ========================================================
    # ASK QWEN
    # ========================================================

    print("AI: ", end="", flush=True)

    stream = chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True
    )


    assistant_message = ""


    # ========================================================
    # STREAM RESPONSE
    # ========================================================

    for chunk in stream:

        text = chunk.message.content

        print(
            text,
            end="",
            flush=True
        )

        assistant_message += text


    print()
    print()


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    messages.append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )


    # ========================================================
    # SAVE MEMORY
    # ========================================================

    save_memory(messages)