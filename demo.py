from src.pipline.training_pipeline import TrainPipeline

pipline = TrainPipeline()
pipline.run_pipeline()

# from pymongo import MongoClient
# from src.constants import DATABASE_NAME, COLLECTION_NAME, MONGODB_URL_KEY
# import os

# client = MongoClient(os.getenv(MONGODB_URL_KEY), serverSelectionTimeoutMS=10000)

# try:
#     client.admin.command("ping")

#     db = client[DATABASE_NAME]
#     print("Database:", DATABASE_NAME)
#     print("Available collections:", db.list_collection_names())

#     collection = db[COLLECTION_NAME]
#     print("Collection:", COLLECTION_NAME)
#     print("Document count:", collection.count_documents({}))
#     print("Sample document:", collection.find_one())
# finally:
#     client.close()