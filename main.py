from ollama import chat
from ddgs import DDGS
from datetime import datetime
import json
import os


MEMORY_FILE = "memory.json"


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

Your name is Odysseus.
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
# INTELLIGENT QUERY ROUTER
# ============================================================

def needs_web_search(text):
    router_prompt = f"""
You are a query router.

Your ONLY job is to decide whether the user's question requires
a web search.

Return ONLY one word:

YES

or

NO

Return YES when the question requires:
- Current or recent information
- News
- Today's information
- Current prices
- Current weather
- Current sports information
- Current political information
- Current products or technology
- Information that may have changed since your training data
- Information about a person, company, product, event, or other
  topic where the user is asking about its current status

Return NO when the question is:
- General knowledge
- Programming help
- Mathematics
- Writing
- Translation
- Casual conversation
- A question that does not require current information

User question:
{text}

Answer:
"""

    response = chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": router_prompt
            }
        ]
    )

    decision = response.message.content.strip().upper()

    print(f"[Router decision: {decision}]")

    return decision.startswith("YES")


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

print("Local AI Chat")
print("Intelligent web search is enabled.")
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
    # ASK ROUTER
    # --------------------------------------------------------

    if needs_web_search(user_input):

        print("[Router] This question may require current information.")

        search_results = web_search(user_input)


        # Add the original user question
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


    else:

        print("[Router] No web search needed.")


        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


    # --------------------------------------------------------
    # ASK QWEN
    # --------------------------------------------------------

    print("AI: ", end="", flush=True)

    stream = chat(
        model="qwen3:4b",
        messages=messages,
        stream=True
    )


    assistant_message = ""


    # --------------------------------------------------------
    # STREAM RESPONSE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    messages.append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )


    # --------------------------------------------------------
    # SAVE MEMORY
    # --------------------------------------------------------

    save_memory(messages)