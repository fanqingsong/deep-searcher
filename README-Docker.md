# Deep Searcher Docker 部署指南

本文档说明如何使用 Docker Compose 部署 Deep Searcher 应用。

## 🚀 快速开始

### 1. 前置要求

- Docker
- Docker Compose
- 至少 4GB 可用内存

### 2. 环境配置

复制环境变量模板文件并配置你的 API 密钥：

```bash
# 复制环境变量模板
cp env.example .env

# 编辑 .env 文件，填入你的 API 密钥
nano .env
```

**重要**: 至少需要配置一个 LLM API 密钥（如 OpenAI）才能正常使用。

### 3. 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f deep-searcher
```

### 4. 服务访问

启动成功后，可以通过以下地址访问各个服务：

- **Deep Searcher API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **Milvus 数据库**: localhost:19530
- **Attu (Milvus Web UI)**: http://localhost:3000
- **MinIO (对象存储)**: http://localhost:9001

## 📋 服务架构

Docker Compose 包含以下服务：

1. **deep-searcher**: 主应用服务
2. **milvus**: Milvus 向量数据库
3. **etcd**: Milvus 的元数据存储
4. **minio**: Milvus 的对象存储
5. **attu**: Milvus 的 Web 管理界面

## 🔧 配置说明

### 向量数据库配置

应用会自动连接到 Docker 网络中的 Milvus 服务：

```yaml
vector_db:
  provider: "Milvus"
  config:
    uri: "http://milvus:19530"  # Docker 服务名
    default_collection: "deepsearcher"
```

### 数据持久化

以下数据会持久化保存：

- `./data`: 应用数据文件
- `./logs`: 应用日志
- Milvus 数据（存储在 Docker volume 中）
- MinIO 数据（存储在 Docker volume 中）
- etcd 数据（存储在 Docker volume 中）

## 🛠️ 常用操作

### 查看服务状态

```bash
# 查看所有服务状态
docker-compose ps

# 查看特定服务日志
docker-compose logs deep-searcher
docker-compose logs milvus
```

### 重启服务

```bash
# 重启所有服务
docker-compose restart

# 重启特定服务
docker-compose restart deep-searcher
```

### 停止服务

```bash
# 停止所有服务
docker-compose down

# 停止服务并删除数据卷（注意：会丢失所有数据）
docker-compose down -v
```

### 更新应用

```bash
# 重新构建并启动应用
docker-compose up -d --build deep-searcher
```

## 📊 使用示例

### 1. 上传文档

```bash
curl -X POST "http://localhost:8000/load-files/" \
  -H "Content-Type: application/json" \
  -d '{
    "paths": ["/app/data/WhatisMilvus.pdf"],
    "collection_name": "test_collection",
    "collection_description": "Test documents"
  }'
```

### 2. 查询文档

```bash
curl -X GET "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -G \
  --data-urlencode "original_query=What is Milvus?" \
  --data-urlencode "max_iter=3"
```

## 🚨 故障排除

### 服务启动失败

1. 检查端口是否被占用：
```bash
netstat -tuln | grep -E "(8000|19530|3000|9000|9001)"
```

2. 检查 Docker 资源：
```bash
docker system df
docker system prune  # 清理不用的资源
```

### Milvus 连接问题

1. 检查 Milvus 健康状态：
```bash
curl http://localhost:9091/healthz
```

2. 查看 Milvus 日志：
```bash
docker-compose logs milvus
```

### 内存不足

如果遇到内存不足问题，可以：

1. 增加 Docker 可用内存
2. 减少同时运行的服务
3. 使用轻量级配置

## 🔐 安全注意事项

1. **API 密钥**: 确保 `.env` 文件不被提交到版本控制
2. **网络访问**: 生产环境建议配置防火墙规则
3. **数据备份**: 定期备份重要的向量数据

## 📝 环境变量说明

主要的环境变量包括：

- `OPENAI_API_KEY`: OpenAI API 密钥
- `DEEPSEEK_API_KEY`: DeepSeek API 密钥  
- `ANTHROPIC_API_KEY`: Anthropic Claude API 密钥
- `FIRECRAWL_API_KEY`: FireCrawl 网页爬虫 API 密钥

更多环境变量请参考 `env.example` 文件。 