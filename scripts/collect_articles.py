import requests
import datetime
import time
import argparse
import json
import os

API_KEY = 'd25fcca640b44ae8b9dbbd4db4820507'


def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)



# Source ids taken from the apis pre-existing list of known North American ids
def load_na_source_ids(json_path="north_american_sources.json"):
    """Load NA source IDs from JSON file into a Python set."""
    data = load_json_file(json_path, [])
    ids = {entry["id"] for entry in data if entry.get("id")}
    print(f"Loaded {len(ids)} North American source IDs.")
    return ids


def normalize_title(title):
    """
    Normalize title for duplicate detection.
    Converts to lowercase and strips whitespace for comparison.
    """
    if not title:
        return ""
    return title.lower().strip()


def fetch_from_api(q, start_date, end_date, page=1):
    url = 'https://newsapi.org/v2/everything'
    
    parameters = {
        'q': q,
        'searchIn': 'title',
        'from': start_date,
        'to': end_date,
        'language': 'en',
        'sortBy': 'popularity',
        'page': page,
        'pageSize': 100,
        'apiKey': API_KEY
    }

    return requests.get(url, params=parameters).json()


def approve_article(article, na_ids, decisions, decisions_path):
    """Return True if article should be included based on hybrid filtering + memory."""
    src = article.get("source", {})
    src_id = src.get("id")
    src_name = src.get("name") or "Unknown Source"

    # Auto-accept known NA source IDs
    if src_id in na_ids:
        print(f"[AUTO] Accepted: {src_name} ({src_id})")
        return True

    # If ID is missing, check memory:
    if src_id is None:

        # If we've already seen and decided on this source:
        if src_name in decisions:
            decision = decisions[src_name]
            print(f"[MEMORY] {src_name}: previously answered '{decision}'")
            return decision == "y"

        # Otherwise ask user if the source is North American
        print(f"\nSource has no ID: {src_name}")
        print(f"URL: {article.get('url')}")
        choice = input("Include this source? (y/n): ").strip().lower()

        # Store the decision
        if choice in ("y", "n"):
            decisions[src_name] = choice
            save_json_file(decisions_path, decisions)
            print(f"[SAVED] Decision: {src_name} → {choice}")

        return choice == "y"

    # Otherwise reject
    print(f"[REJECT] Non-NA source: {src_name} ({src_id})")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("n", type=int)
    args = parser.parse_args()

    # Load NA IDs
    na_ids = load_na_source_ids()

    # Load memory of yes/no decisions for North American-ness
    decisions_path = "source_decisions.json"
    decisions = load_json_file(decisions_path, {})
    print(f"Loaded {len(decisions)} remembered decisions.")

    # Output list and title tracker
    approved = []
    seen_titles = set()  # Track normalized titles to prevent duplicates

    page = 1
    while len(approved) < args.n:

        print(f"\nFetching page {page}...") 
        data = fetch_from_api(args.query, args.start, args.end, page)

        if data.get("status") != "ok":
            print("Error:", data.get("message"))
            break

        articles = data.get("articles", [])
        if not articles:
            print("No more articles.")
            break

        for article in articles:
            if len(approved) >= args.n:
                break
            
            # Check for duplicate title so that we don't get doubles (I think it might still be happening if the title is slightly different)
            title = article.get("title")
            normalized_title = normalize_title(title)
            
            if normalized_title in seen_titles:
                src_name = article.get("source", {}).get("name", "Unknown")
                print(f"[DUPLICATE] Skipping duplicate title from {src_name}: {title}")
                continue
            
            # Check if article passes source filtering
            if approve_article(article, na_ids, decisions, decisions_path):
                approved.append(article)
                seen_titles.add(normalized_title)
                print(f"Accepted total: {len(approved)}")

        page += 1 # Just kidding free plan only lets you take the first page
        time.sleep(1)

    print(f"\nDone. Final approved count: {len(approved)}")
    print(f"Unique titles collected: {len(seen_titles)}")

    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(approved, f, indent=4)


if __name__ == "__main__":
    main()


#Example Call: python collect_articles.py "Zohran Mamdani" 11-12-2025 11-18-2025 100
