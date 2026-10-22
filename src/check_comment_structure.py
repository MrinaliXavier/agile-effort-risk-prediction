from pymongo import MongoClient
import json

client = MongoClient("mongodb://localhost:27017/")
collection = client["JiraReposAnon"]["Spring"]

# XD-1 is a known issue key from our feature pull - check its raw structure
doc = collection.find_one({"key": "XD-1"})

print("Top-level keys:", list(doc.keys()))
print("\nKeys inside 'fields':", list(doc["fields"].keys()))

# Look specifically for anything comment-related
for key in doc["fields"].keys():
    if "comment" in key.lower():
        print(f"\nFound fields.{key}:")
        print(json.dumps(doc["fields"][key], indent=2, default=str)[:2000])

# Also check top-level, in case comments live outside 'fields'
for key in doc.keys():
    if "comment" in key.lower():
        print(f"\nFound top-level '{key}':")
        print(json.dumps(doc[key], indent=2, default=str)[:2000])