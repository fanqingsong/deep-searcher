from fastapi import FastAPI, HTTPException
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files, load_from_website
from deepsearcher.online_query import query
import uvicorn
from config import Settings

app = FastAPI()

settings = Settings()

config = Configuration()

config.set_provider_config(
    "llm",
    settings.llm_provider,
    {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key
    }
)

init_config(config)

@app.post("/load-files/")
def load_files(paths: list[str], collection_name: str = None, collection_description: str = None):
    try:
        load_from_local_files(paths_or_directory=paths, collection_name=collection_name, collection_description=collection_description)
        return {"message": "Files loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-website/")
def load_website(urls: str, collection_name: str = None, collection_description: str = None):
    try:
        load_from_website(urls=urls, collection_name=collection_name, collection_description=collection_description)
        return {"message": "Website loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/")
def perform_query(original_query: str, max_iter: int = 3):
    try:
        result = query(original_query, max_iter)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
