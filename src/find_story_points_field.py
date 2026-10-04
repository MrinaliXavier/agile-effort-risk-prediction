import json

with open(r"D:\Mrinali\FYP\agile-effort-risk-prediction\data\raw\ThePublicJiraDataset\0. DataDefinition\jira_field_information.json", "r", encoding="utf-8") as f:
    field_data = json.load(f)

for repo, fields in field_data.items():
    matches = [f for f in fields if "story point" in f["name"].lower()]
    if matches:
        for m in matches:
            print(f"{repo}: id={m['id']}  name={m['name']}")
    else:
        print(f"{repo}: NO story points field found")