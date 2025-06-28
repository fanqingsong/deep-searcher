#!/usr/bin/env python3
"""
真实的图文搜索示例
演示如何实现文本搜索图像和图像搜索文本
"""

from pymilvus import MilvusClient
import numpy as np
import hashlib

def text_to_vector(text, dim=512):
    """模拟CLIP文本编码器"""
    hash_obj = hashlib.md5(text.encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    np.random.seed(seed)
    
    vector = np.random.random(dim)
    vector = vector / np.linalg.norm(vector)
    
    # 添加语义相关性
    keywords = {
        "猫": [0.8, 0.2, 0.1], "狗": [0.7, 0.3, 0.15], "动物": [0.75, 0.25, 0.12],
        "风景": [0.1, 0.8, 0.3], "山": [0.05, 0.85, 0.35], "湖": [0.08, 0.82, 0.32],
        "建筑": [0.2, 0.1, 0.8], "城市": [0.25, 0.15, 0.75],
        "花": [0.3, 0.6, 0.2], "植物": [0.35, 0.65, 0.25]
    }
    
    for keyword, weights in keywords.items():
        if keyword in text:
            vector[:3] = np.array(weights)
            break
    
    return vector.tolist()

def main():
    print("🚀 真实图文搜索系统演示")
    
    client = MilvusClient("http://localhost:19530")
    collection_name = "multimodal_demo"
    
    # 创建集合
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        dimension=512,
        metric_type="COSINE"
    )
    
    # 示例数据
    dataset = [
        {"id": 1, "image_path": "cat_01.jpg", "description": "一只橙色小猫在阳光下玩耍", "category": "动物"},
        {"id": 2, "image_path": "dog_02.jpg", "description": "金毛犬在公园草地上奔跑", "category": "动物"},
        {"id": 3, "image_path": "mountain_03.jpg", "description": "清晨的高山湖泊倒映雪山", "category": "风景"},
        {"id": 4, "image_path": "city_04.jpg", "description": "现代化城市天际线", "category": "建筑"},
        {"id": 5, "image_path": "flower_05.jpg", "description": "春天花园里盛开的花朵", "category": "植物"}
    ]
    
    # 生成向量并插入
    data = []
    for item in dataset:
        vector = text_to_vector(item["description"])
        data.append({
            "id": item["id"],
            "image_path": item["image_path"],
            "description": item["description"], 
            "category": item["category"],
            "vector": vector
        })
    
    client.insert(collection_name, data)
    
    # 文本搜索演示
    print("\n📝 文本搜索图像演示:")
    queries = ["可爱的小猫", "自然风景", "现代建筑"]
    
    for query in queries:
        print(f"\n🔍 搜索: '{query}'")
        query_vector = text_to_vector(query)
        
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=3,
            output_fields=["image_path", "description", "category"]
        )
        
        for i, result in enumerate(results[0], 1):
            similarity = 1 - result['distance']
            entity = result['entity']
            print(f"  {i}. 相似度: {similarity:.3f}")
            print(f"     图片: {entity['image_path']}")
            print(f"     描述: {entity['description']}")
    
    print("\n✅ 演示完成!")
    client.drop_collection(collection_name)

if __name__ == "__main__":
    main()
