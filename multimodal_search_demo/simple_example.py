#!/usr/bin/env python3

from pymilvus import MilvusClient
import numpy as np

def main():
    # 1. 连接 Milvus (使用Docker服务)
    print("🔗 连接到 Milvus...")
    client = MilvusClient("http://localhost:19530")
    
    collection_name = "image_search"
    
    # 2. 创建集合
    print("📦 创建集合...")
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        dimension=384,
        metric_type="COSINE",
        consistency_level="Strong"
    )
    
    # 3. 插入数据
    print("⬆️ 插入示例数据...")
    data = [
        {"id": 1, "vector": np.random.random(384).tolist(), "text": "可爱的小猫图片", "category": "动物"},
        {"id": 2, "vector": np.random.random(384).tolist(), "text": "金毛犬在草地玩耍", "category": "动物"},
        {"id": 3, "vector": np.random.random(384).tolist(), "text": "美丽的山脉风景", "category": "风景"},
        {"id": 4, "vector": np.random.random(384).tolist(), "text": "现代化城市建筑", "category": "建筑"},
        {"id": 5, "vector": np.random.random(384).tolist(), "text": "彩色花朵盛开", "category": "植物"}
    ]
    
    client.insert(collection_name, data)
    
    # 4. 等待数据插入完成
    print("🔍 等待数据处理完成...")
    import time
    time.sleep(2)  # 简单等待，确保数据插入完成
    
    # 5. 加载集合
    client.load_collection(collection_name)
    
    # 6. 搜索演示
    print("\n" + "="*50)
    print("🔍 搜索演示")
    print("="*50)
    
    # 模拟不同类型的查询向量
    queries = [
        ("动物查询", np.random.seed(1)),
        ("风景查询", np.random.seed(3)), 
        ("建筑查询", np.random.seed(4))
    ]
    
    for query_name, seed in queries:
        print(f"\n📝 {query_name}:")
        np.random.seed(1 if "动物" in query_name else 3)  # 设定种子确保结果一致性
        query_vector = np.random.random(384).tolist()
        
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=3,
            output_fields=["text", "category"]
        )
        
        for i, result in enumerate(results[0], 1):
            similarity = 1 - result['distance']
            entity = result['entity']
            print(f"  {i}. 相似度: {similarity:.3f}")
            print(f"     内容: {entity['text']}")
            print(f"     类别: {entity['category']}")
    
    # 7. 条件搜索演示
    print(f"\n🎯 条件搜索演示 (只搜索动物类别):")
    query_vector = np.random.random(384).tolist()
    
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,
        filter='category == "动物"',  # 使用filter而不是expr
        output_fields=["text", "category"]
    )
    
    for i, result in enumerate(results[0], 1):
        similarity = 1 - result['distance']
        entity = result['entity']
        print(f"  {i}. 相似度: {similarity:.3f}")
        print(f"     内容: {entity['text']}")
        print(f"     类别: {entity['category']}")
    
    print("\n✅ 示例运行完成!")
    print("\n💡 真实应用中:")
    print("   - 使用 CLIP/BERT 等模型生成真实向量")
    print("   - 向量维度通常为 512, 768, 1024")
    print("   - 支持图像和文本的跨模态搜索")
    
    # 清理资源
    client.drop_collection(collection_name)
    print("🧹 清理完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("\n🔧 解决方案:")
        print("1. 确保 Milvus 服务正在运行:")
        print("   docker-compose up -d")
        print("2. 检查端口 19530 是否可访问")
        print("3. 安装依赖: pip install pymilvus numpy") 