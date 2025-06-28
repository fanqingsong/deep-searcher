#!/usr/bin/env python3

from pymilvus import MilvusClient
import numpy as np
import hashlib
import time

def text_to_vector(text, dim=384):
    """将文本转换为向量"""
    hash_obj = hashlib.md5(text.encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    np.random.seed(seed)
    return np.random.random(dim).tolist()

def main():
    print("🚀 图文搜索演示")
    
    # 连接Milvus
    client = MilvusClient("http://localhost:19530")
    collection_name = "demo_search"
    
    # 创建集合
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        dimension=384,
        metric_type="COSINE"
    )
    
    # 插入数据
    data = [
        {"id": 1, "text": "可爱的橙色小猫", "vector": text_to_vector("可爱的橙色小猫")},
        {"id": 2, "text": "金毛犬在草地奔跑", "vector": text_to_vector("金毛犬在草地奔跑")},
        {"id": 3, "text": "高山湖泊美景", "vector": text_to_vector("高山湖泊美景")},
        {"id": 4, "text": "现代城市建筑", "vector": text_to_vector("现代城市建筑")},
        {"id": 5, "text": "花园里的花朵", "vector": text_to_vector("花园里的花朵")}
    ]
    
    client.insert(collection_name, data)
    time.sleep(2)  # 等待数据处理
    
    # 搜索测试
    queries = ["小猫", "风景", "建筑"]
    
    for query in queries:
        print(f"\n🔍 搜索: '{query}'")
        query_vector = text_to_vector(query)
        
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=3,
            output_fields=["text"]
        )
        
        if results and results[0]:
            for i, result in enumerate(results[0], 1):
                similarity = 1 - result['distance']
                print(f"  {i}. 相似度: {similarity:.3f} - {result['entity']['text']}")
        else:
            print("  没有找到结果")
    
    client.drop_collection(collection_name)
    print("\n✅ 演示完成!")

if __name__ == "__main__":
    main()
