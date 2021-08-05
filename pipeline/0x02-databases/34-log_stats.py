#!/usr/bin/env python3
""" Log stats """
from pymongo import MongoClient


if __name__ == "__main__":
    """ Script that provides some stats about Nginx logs stored in MongoDB """

    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx
    numberOfDocuments = collection.count_documents({})
    print("{} logs".format(numberOfDocuments))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    for method in methods:
        nmethod = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, nmethod))

    pathMethod = {"method": "GET", "path": "/status"}
    npath = collection.count_documents(pathMethod)
    print("{} status check".format(npath))
