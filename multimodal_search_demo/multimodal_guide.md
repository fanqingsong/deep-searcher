# Milvus 图文搜索实战教程

## 🎯 目标
构建一个支持文本搜索图像、图像搜索文本的多模态检索系统。

## 📦 环境准备

```bash
# 安装依赖
pip install pymilvus transformers torch pillow requests numpy

# 启动 Milvus (使用 docker-compose)
docker-compose up -d
```

## 🔧 核心步骤

### 1. 连接数据库
```python
from pymilvus import MilvusClient
client = MilvusClient("http://localhost:19530")
```

### 2. 创建集合
```python
# 定义多模态集合结构
collection_schema = {
    "fields": [
        {"name": "id", "type": "INT64", "is_primary": True},
        {"name": "image_path", "type": "VARCHAR", "max_length": 500},
        {"name": "description", "type": "VARCHAR", "max_length": 1000}, 
        {"name": "image_vector", "type": "FLOAT_VECTOR", "dim": 512},
        {"name": "text_vector", "type": "FLOAT_VECTOR", "dim": 512}
    ]
}
```

### 3. 向量化处理
```python
from transformers import CLIPProcessor, CLIPModel

# 加载 CLIP 模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    features = model.get_image_features(**inputs)
    return features.detach().numpy().flatten()

def encode_text(text):
    inputs = processor(text=text, return_tensors="pt")  
    features = model.get_text_features(**inputs)
    return features.detach().numpy().flatten()
```

### 4. 数据插入
```python
# 准备数据
data = []
for image_path, description in image_text_pairs:
    data.append({
        "image_path": image_path,
        "description": description,
        "image_vector": encode_image(image_path).tolist(),
        "text_vector": encode_text(description).tolist()
    })

# 插入向量数据库
client.insert("multimodal_collection", data)
```

### 5. 创建索引
```python
# 为向量字段创建高性能索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}

client.create_index("multimodal_collection", "image_vector", index_params)
client.create_index("multimodal_collection", "text_vector", index_params)
```

### 6. 跨模态搜索
```python
# 文本搜索图像
def text_to_image_search(query_text, top_k=5):
    query_vector = encode_text(query_text)
    results = client.search(
        collection_name="multimodal_collection",
        data=[query_vector.tolist()],
        anns_field="image_vector",  # 用文本向量搜索图像向量
        limit=top_k
    )
    return results

# 图像搜索文本  
def image_to_text_search(query_image, top_k=5):
    query_vector = encode_image(query_image)
    results = client.search(
        collection_name="multimodal_collection", 
        data=[query_vector.tolist()],
        anns_field="text_vector",   # 用图像向量搜索文本向量
        limit=top_k
    )
    return results
```

## 🚀 运行示例

```bash
# 运行简单示例
python simple_example.py

# 运行完整示例  
python multimodal_search_example.py
```

## 💡 应用场景

1. **电商搜索**: 用户上传图片搜索相似商品
2. **内容管理**: 根据文本搜索相关图片素材
3. **智能相册**: 语义化搜索个人照片
4. **推荐系统**: 跨模态内容推荐

## 🎛️ 性能优化

- 选择合适的向量维度 (384/512/768)
- 调整索引参数平衡性能和精度
- 使用 GPU 加速向量计算
- 批量处理提升插入效率 