import requests
import json

API_KEY = "d25fcca640b44ae8b9dbbd4db4820507"

def fetch_sources():
    url = "https://newsapi.org/v2/top-headlines/sources"
    params = {"apiKey": API_KEY}
    resp = requests.get(url, params=params)
    data = resp.json()

    if data.get("status") != "ok":
        print("Error:", data.get("message"))
        return []

    return data.get("sources", [])

def filter_sources(sources, countries=("us", "ca")):
    return [src for src in sources if src.get("country") in countries]

def main():
    sources = fetch_sources()
    na_sources = filter_sources(sources)

    # Extract IDs only
    source_ids = [src["id"] for src in na_sources if src.get("id")]

    # Comma-separated string (for NewsAPI calls)
    sources_string = ",".join(source_ids)

    print("\n=== North American Source IDs (comma-separated) ===\n")
    print(sources_string)

    print("\n=== Pretty list ===\n")
    for src in na_sources:
        print(f"{src['id']:25}  |  {src['name']}")

    # Save full metadata to JSON
    with open("north_american_sources.json", "w", encoding="utf-8") as f:
        json.dump(na_sources, f, indent=4)

    print("\nSaved to north_american_sources.json\n")

if __name__ == "__main__":
    main()
