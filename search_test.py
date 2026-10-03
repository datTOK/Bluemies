from ddgs import DDGS

query = "current president of the USA"

results = DDGS().text(query, max_results=5)

for result in results:
    print("TITLE:", result["title"])
    print("URL:", result["href"])
    print("DESCRIPTION:", result["body"])
    print()