# 🔍 Milvus 多模态图文搜索演示

本目录包含了完整的 Milvus 图文搜索系统实现和演示代码。

## 📁 目录结构

### 🎯 **快速开始示例**
- **`simple_example.py`** - 基础向量搜索演示
- **`demo_search.py`** - 简化搜索演示

### 🚀 **完整多模态系统**
- **`multimodal_search_example.py`** - 完整的CLIP模型实现
- **`real_multimodal_example.py`** - 简化版多模态演示

### 📚 **文档和指南**
- **`README_multimodal_search.md`** - 详细技术文档
- **`multimodal_guide.md`** - 快速入门指南
- **`图文搜索完整教程.md`** - 中文完整教程

### ⚙️ **配置文件**
- **`requirements.txt`** - Python 依赖包列表

## 🏃‍♂️ 快速运行

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 确保 Milvus 服务运行（在上级目录执行）
cd .. && docker-compose up -d
```

### 2. 运行示例
```bash
# 基本向量搜索演示
python3 simple_example.py

# 文本搜索演示
python3 demo_search.py

# 多模态搜索演示
python3 real_multimodal_example.py
```

## 📊 功能对比

| 示例文件 | 复杂度 | 功能 |
|---------|--------|------|
| `simple_example.py` | 简单 | 基础向量搜索 |
| `demo_search.py` | 简单 | 文本相似性搜索 |
| `real_multimodal_example.py` | 中等 | 模拟多模态搜索 |
| `multimodal_search_example.py` | 复杂 | 完整CLIP模型系统 |

## 💡 实际应用场景

- **电商平台**: 用户上传图片搜索相似商品
- **内容管理**: 根据文本描述匹配图片素材
- **智能相册**: 用自然语言搜索个人照片
- **推荐系统**: 跨模态内容推荐

## 🔧 技术栈

- **向量数据库**: Milvus 2.4+
- **多模态模型**: OpenAI CLIP
- **Python 库**: PyMilvus, Transformers, PyTorch
- **容器化**: Docker Compose
