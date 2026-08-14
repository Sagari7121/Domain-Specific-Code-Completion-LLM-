import os

SOURCE_DIR = "data/realworld/backend"
OUTPUT_FILE = "data/input.txt"


def collect_files():
    files = []

    for root, dirs, filenames in os.walk(SOURCE_DIR):

        for filename in filenames:

            if filename.endswith(".js"):
                files.append(os.path.join(root, filename))

    return sorted(files)


def build_dataset():

    files = collect_files()

    print(f"Found {len(files)} JavaScript files")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for file_path in files:

            relative_path = os.path.relpath(file_path, SOURCE_DIR)

            out.write(f"\n// ===== FILE: {relative_path} =====\n\n")

            with open(file_path, "r", encoding="utf-8") as f:
                out.write(f.read())

            out.write("\n\n")

    print(f"Dataset written to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()