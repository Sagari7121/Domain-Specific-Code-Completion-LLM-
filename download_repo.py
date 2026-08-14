import os
import requests

OWNER = "briefercloud"
REPO = "realworld-example-app"
BRANCH = "main"

# We only want these parts of backend
ALLOWED_PATHS = [
    "backend/controllers",
    "backend/helper",
    "backend/middleware",
    "backend/models",
    "backend/routes",
    "backend/index.js",
]

OUTPUT_DIR = "data/realworld"


def get_repo_tree():
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"

    response = requests.get(url)
    response.raise_for_status()

    return response.json()["tree"]

def is_allowed(path):
    for allowed in ALLOWED_PATHS:

        # Directory
        if path.startswith(allowed + "/"):
            return True

        # Exact file
        if path == allowed:
            return True

    return False

def download_file(path):
    url = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{path}"

    response = requests.get(url)
    response.raise_for_status()

    output_path = os.path.join(OUTPUT_DIR, path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Downloaded: {path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tree = get_repo_tree()

    for item in tree:

        if item["type"] != "blob":
            continue

        path = item["path"]

        if not is_allowed(path):
            continue

        # Only download JavaScript files
        if not path.endswith(".js"):
            continue

        download_file(path)


if __name__ == "__main__":
    main()