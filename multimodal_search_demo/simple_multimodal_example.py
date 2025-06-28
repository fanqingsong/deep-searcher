#!/usr/bin/env python3
"""
简化版 Milvus 图文搜索示例
演示如何构建多模态向量数据库并进行搜索
"""

from pymilvus import MilvusClient
import numpy as np

def create_simple_example():
    """创建简单的图文搜索示例"""
    
    # 1. 连接 Milvus (使用 Milvus Lite 本地文件数据库)
    print("🔗 连接到 Milvus...")
    client = MilvusClient("./multimodal_demo.db")
    
    collection_name = "image_text_search"
    
    # 2. 创建集合
    print("📦 创建集合...")
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    # 创建集合 - 定义字段结构
    client.create_collection(
        collection_name=collection_name,
        dimension=384,  # 向量维度
        metric_type="COSINE",  # 余弦相似度
        consistency_level="Strong"
    )
    
    # 3. 准备示例数据 (模拟向量数据)
    print("📊 准备示例数据...")
    
    # 模拟图像和文本的向量表示
    def create_mock_vector(seed):
        """创建模拟向量"""
        np.random.seed(seed)
        return np.random.random(384).tolist()
    
    # 示例数据：图像路径、描述和对应的向量
    sample_data = [
        {
            "id": 1,
            "image_path": "cat_001.jpg",
            "description": "一只可爱的橙色小猫",
            "vector": create_mock_vector(1)
        },
        {
            "id": 2,
            "image_path": "dog_001.jpg", 
            "description": "金毛犬在公园玩耍",
            "vector": create_mock_vector(2)
        },
        {
            "id": 3,
            "image_path": "landscape_001.jpg",
            "description": "美丽的山脉湖泊风景",
            "vector": create_mock_vector(3)
        },
        {
            "id": 4,
            "image_path": "city_001.jpg",
            "description": "现代化城市天际线",
            "vector": create_mock_vector(4)
        },
        {
            "id": 5,
            "image_path": "flowers_001.jpg",
            "description": "花园里的彩色花朵",
            "vector": create_mock_vector(5)
        }
    ]
    
    # 4. 插入数据
    print("⬆️ 插入数据到向量数据库...")
    client.insert(collection_name, sample_data)
    
    # 5. 创建索引 (提升搜索性能)
    print("🔍 创建向量索引...")
    client.create_index(
        collection_name=collection_name,
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )
    
    # 6. 加载集合到内存
    client.load_collection(collection_name)
    
    print("✅ 数据库构建完成!")
    return client, collection_name

def search_examples(client, collection_name):
    """搜索示例演示"""
    
    print("\n" + "="*60)
    print("🔍 搜索演示")
    print("="*60)
    
    # 模拟查询向量 (实际应用中这些是从文本或图像生成的)
    def create_query_vector(seed):
        np.random.seed(seed)
        return np.random.random(384).tolist()
    
    # 搜索示例 1: 查找相似内容
    print("\n📝 示例 1: 搜索 '动物' 相关内容")
    query_vector = create_query_vector(1)  # 模拟 "动物" 的向量
    
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        output_fields=["image_path", "description"]
    )
    
    for i, result in enumerate(results[0], 1):
        similarity = 1 - result['distance']  # 转换为相似度
        print(f"  {i}. 相似度: {similarity:.3f}")
        print(f"     图片: {result['entity']['image_path']}")
        print(f"     描述: {result['entity']['description']}")
    
    # 搜索示例 2: 查找风景图片
    print("\n🏔️ 示例 2: 搜索 '风景' 相关内容")
    query_vector = create_query_vector(3)  # 模拟 "风景" 的向量
    
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        output_fields=["image_path", "description"]
    )
    
    for i, result in enumerate(results[0], 1):
        similarity = 1 - result['distance']
        print(f"  {i}. 相似度: {similarity:.3f}")
        print(f"     图片: {result['entity']['image_path']}")
        print(f"     描述: {result['entity']['description']}")
    
    # 搜索示例 3: 使用筛选条件
    print("\n🔍 示例 3: 带条件筛选的搜索")
    query_vector = create_query_vector(10)
    
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,
        output_fields=["image_path", "description"],
        # 可以添加过滤条件，例如: expr="id > 2"
    )
    
    print("   搜索所有内容 (按相似度排序):")
    for i, result in enumerate(results[0], 1):
        similarity = 1 - result['distance']
        print(f"  {i}. 相似度: {similarity:.3f}")
        print(f"     图片: {result['entity']['image_path']}")
        print(f"     描述: {result['entity']['description']}")

def main():
    """主函数"""
    print("🚀 Milvus 多模态搜索示例启动")
    print("="*60)
    
    try:
        # 构建向量数据库
        client, collection_name = create_simple_example()
        
        # 执行搜索演示
        search_examples(client, collection_name)
        
        print("\n✨ 示例运行完成!")
        print("\n💡 在实际应用中:")
        print("   - 使用 CLIP 等模型将图像/文本转换为向量")
        print("   - 向量维度通常是 512, 768, 1024 等")
        print("   - 可以存储更多元数据字段")
        print("   - 支持实时更新和删除")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("请确保:")
        print("1. 已安装 pymilvus: pip install pymilvus")
        print("2. 当前目录有写权限 (用于创建本地数据库文件)")

if __name__ == "__main__":
    main() 