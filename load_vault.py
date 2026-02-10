from pathlib import Path
from llama_index.readers.obsidian import ObsidianReader
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = Path("/Users/aidanwu/Library/CloudStorage/GoogleDrive-samuraishibe1@gmail.com/My Drive/Aidan's Vault")

reader = ObsidianReader(input_dir=str(VAULT_PATH))
docs = reader.load_data()

print(f"Loaded {len(docs)} notes")
print("Example metadata:", docs[0].metadata)
print("Example preview:", docs[0].text[:300])