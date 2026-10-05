from ollama import chat
from ddgs import DDGS
from datetime import datetime
import json
import os
import ast
import operator


MEMORY_FILE = "memory.json"
MODEL_NAME = "qwen3:4b"

# Only files inside this directory can be accessed by Bluemies.
FILES_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "files"
)


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
- Do not replace newer web information with older information from your training data.
- Pay attention to dates in the search results.
- If the results conflict with your previous knowledge, prefer newer reliable web sources.
- Do not invent information that is not supported by the search results.

When file contents are provided:
- Treat the file contents as the source of truth.
- Do not claim to have accessed files that were not provided.
- Do not invent file contents.

Do not claim that you searched the web unless search results were actually provided.
Do not claim that you accessed a file unless file contents were actually provided.
"""


# ============================================================
# MEMORY
# ============================================================

def load_memory():

    if os.path.exists(MEMORY_FILE):

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


def save_memory(messages):

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

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
- Understand a file
- Inspect a document
- Work with a local file
- Ask about the contents of a local file

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

    decision = decision.replace(
        ".",
        ""
    ).replace(
        ":",
        ""
    ).strip()

    valid_routes = {
        "CHAT",
        "WEB",
        "CALCULATE",
        "CODE",
        "FILE"
    }

    if decision in valid_routes:

        return decision

    print(
        f"[Router] Unexpected decision: {decision}"
    )

    print(
        "[Router] Falling back to CHAT."
    )

    return "CHAT"


# ============================================================
# CALCULATOR
# ============================================================

def calculate(expression):

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):

        if isinstance(
            node,
            ast.Constant
        ):

            if isinstance(
                node.value,
                (int, float)
            ):

                return node.value

            raise ValueError(
                "Invalid value."
            )

        if isinstance(
            node,
            ast.BinOp
        ):

            left = evaluate(
                node.left
            )

            right = evaluate(
                node.right
            )

            operator_function = allowed_operators.get(
                type(node.op)
            )

            if operator_function is None:

                raise ValueError(
                    "Operator not allowed."
                )

            return operator_function(
                left,
                right
            )

        if isinstance(
            node,
            ast.UnaryOp
        ):

            operand = evaluate(
                node.operand
            )

            operator_function = allowed_operators.get(
                type(node.op)
            )

            if operator_function is None:

                raise ValueError(
                    "Operator not allowed."
                )

            return operator_function(
                operand
            )

        raise ValueError(
            "Invalid expression."
        )

    tree = ast.parse(
        expression,
        mode="eval"
    )

    return evaluate(
        tree.body
    )


# ============================================================
# CLEAN CALCULATOR INPUT
# ============================================================

def clean_calculation_expression(text):

    expression = text.strip()

    prefixes = [
        "calculate ",
        "what is ",
        "what's ",
        "compute ",
        "solve "
    ]

    expression_lower = expression.lower()

    for prefix in prefixes:

        if expression_lower.startswith(
            prefix
        ):

            expression = expression[
                len(prefix):
            ].strip()

            break

    expression = expression.rstrip(
        "?.!"
    ).strip()

    return expression


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query):

    print(
        "\n[Tool] Searching the web..."
    )

    try:

        results = DDGS().text(
            query,
            max_results=5
        )

    except Exception as error:

        print(
            f"[Tool] Search failed: {error}"
        )

        return "Web search failed."

    if not results:

        return "No search results found."

    formatted_results = []

    for i, result in enumerate(
        results,
        start=1
    ):

        formatted_results.append(
            f"""
Result {i}
Title: {result["title"]}
URL: {result["href"]}
Description: {result["body"]}
"""
        )

    return "\n".join(
        formatted_results
    )


# ============================================================
# FILE TOOL
# ============================================================

def find_file(filename):

    """
    Find a file inside the allowed files directory.
    """

    filename = filename.strip()

    # Prevent absolute paths
    if os.path.isabs(filename):

        return None

    # Prevent path traversal such as ../
    safe_path = os.path.abspath(
        os.path.join(
            FILES_DIRECTORY,
            filename
        )
    )

    allowed_directory = os.path.abspath(
        FILES_DIRECTORY
    )

    if not safe_path.startswith(
        allowed_directory + os.sep
    ):

        return None

    if not os.path.isfile(
        safe_path
    ):

        return None

    return safe_path


def read_file(filename):

    """
    Read a text file from the allowed files directory.
    """

    file_path = find_file(
        filename
    )

    if file_path is None:

        return None

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    except UnicodeDecodeError:

        return None

    except OSError:

        return None


def list_files():

    """
    List files available inside the allowed directory.
    """

    if not os.path.exists(
        FILES_DIRECTORY
    ):

        os.makedirs(
            FILES_DIRECTORY
        )

    files = []

    for root, directories, filenames in os.walk(
        FILES_DIRECTORY
    ):

        for filename in filenames:

            full_path = os.path.join(
                root,
                filename
            )

            relative_path = os.path.relpath(
                full_path,
                FILES_DIRECTORY
            )

            files.append(
                relative_path
            )

    return files


def extract_filename(text):

    """
    Try to extract a filename from the user's request.
    """

    text = text.strip()

    # First, look for a filename with a common extension.
    words = text.replace(
        '"',
        ""
    ).replace(
        "'",
        ""
    ).split()

    common_extensions = (
        ".txt",
        ".md",
        ".py",
        ".json",
        ".csv",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".tsx",
        ".jsx"
    )

    for word in words:

        cleaned_word = word.strip(
            ".,?!:;"
        )

        if cleaned_word.lower().endswith(
            common_extensions
        ):

            return cleaned_word

    return None


# ============================================================
# START MEMORY
# ============================================================

messages = load_memory()


# ============================================================
# PROGRAM START
# ============================================================

print(
    "Bluemies - Local AI"
)

print(
    "Intelligent tool routing is enabled."
)

print(
    "Type 'exit' to quit."
)

print()


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    user_input = input(
        "You: "
    )


    # ========================================================
    # EXIT
    # ========================================================

    if user_input.lower() == "exit":

        save_memory(
            messages
        )

        print(
            "Conversation saved."
        )

        print(
            "Goodbye!"
        )

        break


    # ========================================================
    # ROUTE THE REQUEST
    # ========================================================

    route = route_request(
        user_input
    )

    print(
        f"[Router] Selected route: {route}"
    )


    # ========================================================
    # WEB ROUTE
    # ========================================================

    if route == "WEB":

        print(
            "[Router] This question requires current information."
        )

        search_results = web_search(
            user_input
        )

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

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
- Pay attention to dates in the search results.
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

        print(
            "[Router] Calculation detected."
        )

        print(
            "[Tool] Calculator is running..."
        )

        try:

            expression = clean_calculation_expression(
                user_input
            )

            print(
                f"[Tool] Expression: {expression}"
            )

            result = calculate(
                expression
            )

            print(
                f"[Tool] Result: {result}"
            )

            messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": f"""
A calculator tool was used to calculate the user's expression.

Expression:
{expression}

Calculator result:
{result}

Give the user a concise answer using this result.

Important:
- Do not perform the calculation again yourself.
- Do not change the calculator result.
- Do not claim that a different result was obtained.
"""
                }
            )

        except Exception as error:

            print(
                f"[Tool] Calculator error: {error}"
            )

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
The calculator could not understand the user's expression.

Tell the user that the expression could not be calculated
and ask them to provide a clearer mathematical expression.

Do not invent a result.
"""
                }
            )


    # ========================================================
    # FILE ROUTE
    # ========================================================

    elif route == "FILE":

        print(
            "[Router] File operation detected."
        )

        filename = extract_filename(
            user_input
        )


        # ----------------------------------------------------
        # No filename found
        # ----------------------------------------------------

        if filename is None:

            print(
                "[Tool] No filename detected."
            )

            available_files = list_files()

            if available_files:

                files_text = "\n".join(
                    f"- {file}"
                    for file in available_files
                )

            else:

                files_text = (
                    "No files are currently available."
                )

            messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": f"""
The user requested a file operation, but no specific
filename could be identified.

The files currently available inside the allowed files
directory are:

{files_text}

Ask the user which file they want to work with.
"""
                }
            )


        # ----------------------------------------------------
        # Filename found
        # ----------------------------------------------------

        else:

            print(
                f"[Tool] Requested file: {filename}"
            )

            file_contents = read_file(
                filename
            )


            # ------------------------------------------------
            # File not found
            # ------------------------------------------------

            if file_contents is None:

                print(
                    "[Tool] File could not be read."
                )

                available_files = list_files()

                if available_files:

                    files_text = "\n".join(
                        f"- {file}"
                        for file in available_files
                    )

                else:

                    files_text = (
                        "No files are currently available."
                    )

                messages.append(
                    {
                        "role": "user",
                        "content": user_input
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": f"""
The requested file could not be found or read.

Requested file:
{filename}

Available files:

{files_text}

Tell the user that the file could not be found or read.
Do not invent its contents.
"""
                    }
                )


            # ------------------------------------------------
            # File successfully read
            # ------------------------------------------------

            else:

                print(
                    "[Tool] File read successfully."
                )

                messages.append(
                    {
                        "role": "user",
                        "content": user_input
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": f"""
A file tool successfully read the following file.

Filename:
{filename}

File contents:
--------------------
{file_contents}
--------------------

Use the file contents to answer the user's original question.

Important:
- The file contents are the source of truth.
- Do not invent information that is not present in the file.
- Do not claim that you accessed any other files.
"""
                    }
                )


    # ========================================================
    # CODE ROUTE
    # ========================================================

    elif route == "CODE":

        print(
            "[Router] Programming request detected."
        )

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
    # CHAT ROUTE
    # ========================================================

    else:

        print(
            "[Router] Normal conversation."
        )

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


    # ========================================================
    # ASK QWEN
    # ========================================================

    print(
        "AI: ",
        end="",
        flush=True
    )

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

    save_memory(
        messages
    )