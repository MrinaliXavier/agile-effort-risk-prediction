# # from pymongo import MongoClient
# # from pprint import pprint

# # client = MongoClient("mongodb://localhost:27017/")

# # print("Databases:")
# # for db_name in client.list_database_names():
# #     print(" -", db_name)



# # from pymongo import MongoClient
# # from pprint import pprint

# # client = MongoClient("mongodb://localhost:27017/")
# # db = client["JiraReposAnon"]

# # print("Collections:")
# # for name in db.list_collection_names():
# #     print(" -", name)

# # from pymongo import MongoClient
# # from pprint import pprint

# # client = MongoClient("mongodb://localhost:27017/")
# # db = client["JiraReposAnon"]

# # # pick Spring since we know its Story Points field already
# # collection = db["Spring"]

# # print("Document count:", collection.count_documents({}))

# # one_issue = collection.find_one()
# # pprint(one_issue)


# # from pymongo import MongoClient
# # import pandas as pd

# # client = MongoClient("mongodb://localhost:27017/")
# # db = client["JiraReposAnon"]
# # collection = db["Spring"]

# # STORY_POINTS_FIELDS = ["customfield_10142", "customfield_10781"]  # from our earlier lookup

# # # 1. What projects exist inside this collection, and how many issues each has?
# # pipeline = [
# #     {"$group": {"_id": "$fields.project.key", "count": {"$sum": 1}}},
# #     {"$sort": {"count": -1}}
# # ]
# # print("Projects inside 'Spring' collection:")
# # for row in collection.aggregate(pipeline):
# #     print(f"  {row['_id']}: {row['count']} issues")

# # # 2. Of all issues, how many actually have a Story Points value filled in?
# # for sp_field in STORY_POINTS_FIELDS:
# #     count = collection.count_documents({f"fields.{sp_field}": {"$ne": None}})
# #     print(f"\nIssues with {sp_field} populated: {count}")

# # # 3. Issue type breakdown (are these mostly Stories, or mostly Bugs like our sample?)
# # pipeline2 = [
# #     {"$group": {"_id": "$fields.issuetype.name", "count": {"$sum": 1}}},
# #     {"$sort": {"count": -1}}
# # ]
# # print("\nIssue type breakdown:")
# # for row in collection.aggregate(pipeline2):
# #     print(f"  {row['_id']}: {row['count']}")

# # #
# # This will show us exactly which project(s) within Spring are the best candidates — ideally we want a project with a few hundred+ story-pointed issues, since that's the minimum you'd realistically want for training a regression model.#

# from pymongo import MongoClient

# client = MongoClient("mongodb://localhost:27017/")
# db = client["JiraReposAnon"]
# collection = db["Spring"]

# pipeline = [
#     {"$match": {"fields.customfield_10142": {"$ne": None}}},
#     {"$group": {"_id": "$fields.project.key", "count": {"$sum": 1}}},
#     {"$sort": {"count": -1}}
# ]

# print("Projects with Story Points (customfield_10142) populated:")
# for row in collection.aggregate(pipeline):
#     print(f"  {row['_id']}: {row['count']} issues with story points")

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["JiraReposAnon"]
collection = db["Spring"]

xd_filter = {"fields.project.key": "XD"}

print("Total XD issues:", collection.count_documents(xd_filter))
print("XD issues with Story Points:", collection.count_documents({**xd_filter, "fields.customfield_10142": {"$ne": None}}))
print("XD issues with resolutiondate set:", collection.count_documents({**xd_filter, "fields.resolutiondate": {"$ne": None}}))
print("XD issues with timeoriginalestimate set:", collection.count_documents({**xd_filter, "fields.timeoriginalestimate": {"$ne": None}}))
print("XD issues with timespent set:", collection.count_documents({**xd_filter, "fields.timespent": {"$ne": None}}))

print("\nXD issue type breakdown:")
pipeline = [
    {"$match": xd_filter},
    {"$group": {"_id": "$fields.issuetype.name", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
for row in collection.aggregate(pipeline):
    print(f"  {row['_id']}: {row['count']}")

print("\nXD status breakdown (useful for 'risk' definitions like reopened):")
pipeline2 = [
    {"$match": xd_filter},
    {"$group": {"_id": "$fields.status.name", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
for row in collection.aggregate(pipeline2):
    print(f"  {row['_id']}: {row['count']}")