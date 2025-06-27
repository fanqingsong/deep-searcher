import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))

config = Configuration()  # Customize your config here

# Use SiliconFlow instead of OpenAI since we have the API key available
config.set_provider_config("llm", "SiliconFlow", {"model": "deepseek-ai/DeepSeek-V3"})
config.set_provider_config("embedding", "SiliconflowEmbedding", {"model": "BAAI/bge-m3"})

# Use Qdrant in memory mode to avoid external dependencies
# config.set_provider_config("vector_db", "Qdrant", {
#     "default_collection": "deepsearcher",
#     "location": ":memory:"
# })

# Use Milvus service from docker-compose
config.set_provider_config("vector_db", "Milvus", {
    "default_collection": "deepsearcher",
    "uri": "http://localhost:19530",
    "token": "root:Milvus",
    "db": "default"
})

init_config(config=config)


# You should clone the milvus docs repo to your local machine first, execute:
# git clone https://github.com/milvus-io/milvus-docs.git
# Then replace the path below with the path to the milvus-docs repo on your local machine
# import glob
# all_md_files = glob.glob('xxx/milvus-docs/site/en/**/*.md', recursive=True)
# load_from_local_files(paths_or_directory=all_md_files, collection_name="milvus_docs", collection_description="All Milvus Documents")

# Hint: You can also load a single file, please execute it in the root directory of the deep searcher project
load_from_local_files(
    paths_or_directory=os.path.join(current_dir, "data/WhatisMilvus.pdf"),
    collection_name="milvus_docs",
    collection_description="All Milvus Documents",
    # force_new_collection=True, # If you want to drop origin collection and create a new collection every time, set force_new_collection to True
)

question = "Write a report comparing Milvus with other vector databases."

_, _, consumed_token = query(question, max_iter=1)
print(f"Consumed tokens: {consumed_token}")
