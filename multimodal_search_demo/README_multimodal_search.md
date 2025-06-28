# Milvus 图文搜索完整教程

本教程展示如何使用 Milvus 构建多模态向量数据库，实现图像和文本之间的跨模态搜索。

## 🎯 功能特点

- **文本搜索图像**: 输入文本描述，找到相关图片
- **图像搜索文本**: 输入图片，找到相关文本描述  
- **图像搜索图像**: 找到视觉上相似的图片
- **混合搜索**: 同时使用文本和图像进行搜索

## 📋 环境准备

### 1. 安装依赖包

```bash
pip install -r requirements.txt
```

### 2. 启动 Milvus 服务

**方式一: 使用 Docker Compose (推荐)**
```bash
# 使用项目中的 docker-compose.yml
docker-compose up -d
```

**方式二: 使用 Milvus Lite (本地文件数据库)**
```bash
# 无需额外安装，直接使用文件数据库
# 适合开发和测试
```

## 🚀 快速开始

### 1. 简单示例

```bash
python simple_example.py
```

这个示例展示基本的向量搜索原理。

### 2. 完整多模态示例

```bash
python multimodal_search_example.py
```

这个示例使用 CLIP 模型实现真正的图文搜索。

## 🔧 核心技术原理

### 1. 多模态向量生成

```python
from transformers import CLIPProcessor, CLIPModel

# 加载 CLIP 模型 
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图像向量化
def encode_image(image):
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy()

# 文本向量化  
def encode_text(text):
    inputs = processor(text=text, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy()
```

### 2. 向量数据库构建

```python
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

# 定义数据库结构
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
]

schema = CollectionSchema(fields=fields, description="多模态搜索集合")

# 创建集合
client.create_collection(collection_name="multimodal_search", schema=schema)
```

### 3. 创建向量索引

```python
# 为图像向量创建 HNSW 索引
image_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE", 
    "params": {"M": 16, "efConstruction": 200}
}

client.create_index(
    collection_name="multimodal_search",
    field_name="image_vector",
    index_params=image_index_params
)
```

### 4. 数据插入

```python
# 准备数据
data = []
for image_path, description in image_text_pairs:
    # 生成向量
    image_vector = encode_image(load_image(image_path))
    text_vector = encode_text(description)
    
    data.append({
        "image_path": image_path,
        "description": description,
        "image_vector": image_vector.tolist(),
        "text_vector": text_vector.tolist()
    })

# 批量插入
client.insert("multimodal_search", data)
```

### 5. 跨模态搜索

```python
# 文本搜索图像
def search_images_by_text(query_text, top_k=5):
    # 将文本转换为向量
    query_vector = encode_text(query_text)
    
    # 搜索相似的图像向量
    results = client.search(
        collection_name="multimodal_search",
        data=[query_vector.tolist()],
        anns_field="image_vector",  # 搜索图像向量字段
        limit=top_k,
        output_fields=["image_path", "description"]
    )
    return results

# 图像搜索文本
def search_texts_by_image(query_image_path, top_k=5):
    # 将图像转换为向量
    query_vector = encode_image(load_image(query_image_path))
    
    # 搜索相似的文本向量
    results = client.search(
        collection_name="multimodal_search", 
        data=[query_vector.tolist()],
        anns_field="text_vector",  # 搜索文本向量字段
        limit=top_k,
        output_fields=["image_path", "description"]
    )
    return results
```

## 📊 实际应用示例

### 1. 电商图片搜索

```python
# 用户上传图片，搜索相似商品
results = search_images_by_image("user_upload.jpg", top_k=10)

# 用户输入描述，搜索相关商品
results = search_images_by_text("红色连衣裙", top_k=10)
```

### 2. 内容管理系统

```python
# 根据文章内容搜索相关图片
results = search_images_by_text("科技创新发展", top_k=5)

# 根据图片搜索相关文章
results = search_texts_by_image("tech_image.jpg", top_k=5)
```

### 3. 智能相册

```python
# 语义搜索照片
results = search_images_by_text("生日聚会", top_k=20)
results = search_images_by_text("海边风景", top_k=20)
```

## 🎛️ 高级配置

### 1. 索引参数调优

```python
# HNSW 索引参数
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,              # 连接数，影响索引质量和大小
        "efConstruction": 200  # 构建时搜索范围，影响索引质量
    }
}

# 搜索参数  
search_params = {
    "metric_type": "COSINE",
    "params": {
        "ef": 100  # 搜索时范围，影响召回率和性能
    }
}
```

### 2. 批量处理优化

```python
# 批量插入数据
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    client.insert("multimodal_search", batch)
    
    # 定期刷新确保数据可搜索
    if i % (batch_size * 10) == 0:
        client.flush("multimodal_search")
```

### 3. 内存管理

```python
# 加载集合到内存
client.load_collection("multimodal_search")

# 释放集合内存
client.release_collection("multimodal_search")
```

## 📈 性能优化建议

### 1. 硬件配置

- **CPU**: 16+ 核心，支持 AVX2 指令集
- **内存**: 32GB+ (存储向量数据)
- **GPU**: NVIDIA GPU (加速向量计算)
- **存储**: SSD (提升 I/O 性能)

### 2. 参数调优

```python
# 索引参数权衡
# M 越大 → 质量越好，内存占用越大
# efConstruction 越大 → 质量越好，构建时间越长
# ef 越大 → 召回率越高，搜索时间越长

# 推荐配置
small_dataset = {"M": 8, "efConstruction": 100}   # < 100万向量
medium_dataset = {"M": 16, "efConstruction": 200} # 100万-1000万向量  
large_dataset = {"M": 32, "efConstruction": 400}  # > 1000万向量
```

### 3. 向量维度选择

```python
# 维度与性能权衡
dimensions = {
    384: "平衡性能和效果",    # CLIP ViT-B/32
    512: "较好效果",         # CLIP ViT-B/16  
    768: "更好效果",         # CLIP ViT-L/14
    1024: "最佳效果"         # 大型模型
}
```

## 🔍 故障排除

### 1. 常见错误

```bash
# 连接失败
Error: failed to connect to Milvus
解决: 检查 Milvus 服务是否启动，端口是否正确

# 内存不足
Error: insufficient memory  
解决: 增加内存或减少加载的数据量

# 维度不匹配
Error: dimension mismatch
解决: 确保查询向量维度与集合定义一致
```

### 2. 性能问题

```python
# 搜索慢
# 解决方案:
1. 检查是否创建了索引
2. 调整搜索参数 ef 值
3. 确保集合已加载到内存

# 插入慢  
# 解决方案:
1. 使用批量插入
2. 减少向量维度
3. 优化网络连接
```

## 📚 扩展阅读

- [Milvus 官方文档](https://milvus.io/docs)
- [CLIP 模型详解](https://github.com/openai/CLIP)
- [向量数据库原理](https://zilliz.com/learn/what-is-vector-database)
- [多模态 AI 应用](https://zilliz.com/blog)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个示例！

## �� 许可证

MIT License 