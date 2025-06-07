# Embedding Model Configuration

DeepSearcher converts text into vector embeddings before storing them in the vector database. Two embedding options are built in: **OpenAIEmbedding** and **OllamaEmbedding**.

## 📝 Basic Configuration

```python
config.set_provider_config("embedding", "(EmbeddingModelName)", {"model": "<model>"})
```

Where `EmbeddingModelName` is either `OpenAIEmbedding` or `OllamaEmbedding`.

### OpenAIEmbedding

```python
config.set_provider_config("embedding", "OpenAIEmbedding", {"model": "text-embedding-ada-002"})
```
*Requires `OPENAI_API_KEY` environment variable*

### OllamaEmbedding

```python
config.set_provider_config("embedding", "OllamaEmbedding", {"model": "bge-m3"})
```
*Requires a running local Ollama server*
