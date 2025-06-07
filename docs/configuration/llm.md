# LLM Configuration

DeepSearcher uses large language models to process queries and generate responses. The current version ships with two built-in integrations: **OpenAI** and **Ollama**.

## 📝 Basic Configuration

```python
config.set_provider_config("llm", "(LLMName)", {"model": "<model-name>"})
```

Where `LLMName` is either `OpenAI` or `Ollama`.

### OpenAI

```python
config.set_provider_config("llm", "OpenAI", {"model": "o1-mini"})
```
*Requires `OPENAI_API_KEY` environment variable*

### Ollama

```python
config.set_provider_config("llm", "Ollama", {"model": "qwq"})
```
*Requires a running local Ollama server (default `http://localhost:11434`)*
