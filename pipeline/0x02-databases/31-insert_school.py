#!/usr/bin/env python3
"""  Insert a document """


def insert_school(mongo_collection, **kwargs):
    """Method that inserts a new document in a collection based on kwargs """

    return mongo_collection.insert_one(kwargs).inserted_id
