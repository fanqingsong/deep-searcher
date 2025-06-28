#!/usr/bin/env python3
"""
Milvus 图文搜索完整示例
支持文本搜索图像和图像搜索文本的多模态检索系统
"""

import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalSearchSystem:
    def __init__(self, milvus_uri="http://localhost:19530"):
        """
        初始化多模态搜索系统
        
        Args:
            milvus_uri: Milvus服务器地址
        """
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = "multimodal_search"
        
        # 初始化CLIP模型 - 用于生成图像和文本的向量表示
        logger.info("加载 CLIP 模型...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logger.info(f"使用设备: {self.device}")
        
    def create_collection(self):
        """创建多模态搜索集合"""
        # 检查集合是否已存在
        if self.client.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在，删除后重新创建")
            self.client.drop_collection(self.collection_name)
        
        # 定义集合schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="image_description", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512),  # CLIP向量维度
            FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=512),   # CLIP向量维度
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="多模态图文搜索集合"
        )
        
        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )
        
        logger.info(f"集合 {self.collection_name} 创建成功")
        
    def create_indexes(self):
        """为向量字段创建索引"""
        # 为图像向量创建索引
        image_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        self.client.create_index(
            collection_name=self.collection_name,
            field_name="image_vector",
            index_params=image_index_params
        )
        
        # 为文本向量创建索引
        text_index_params = {
            "index_type": "HNSW", 
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        self.client.create_index(
            collection_name=self.collection_name,
            field_name="text_vector",
            index_params=text_index_params
        )
        
        logger.info("索引创建成功")
        
    def encode_image(self, image_path_or_url):
        """
        将图像编码为向量
        
        Args:
            image_path_or_url: 图像路径或URL
            
        Returns:
            numpy.ndarray: 图像向量
        """
        try:
            # 加载图像
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path_or_url).convert('RGB')
            
            # 处理图像并生成向量
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # 归一化向量
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"图像编码失败 {image_path_or_url}: {e}")
            return None
    
    def encode_text(self, text):
        """
        将文本编码为向量
        
        Args:
            text: 文本内容
            
        Returns:
            numpy.ndarray: 文本向量
        """
        try:
            # 处理文本并生成向量
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # 归一化向量
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"文本编码失败 {text}: {e}")
            return None
    
    def insert_data(self, data_list):
        """
        插入图像和文本数据
        
        Args:
            data_list: 数据列表，每个元素包含 {'image_path': '', 'description': ''}
        """
        entities = []
        
        for data in data_list:
            image_path = data['image_path']
            description = data['description']
            
            # 生成图像向量
            image_vector = self.encode_image(image_path)
            if image_vector is None:
                continue
                
            # 生成文本向量
            text_vector = self.encode_text(description)
            if text_vector is None:
                continue
            
            entities.append({
                "image_path": image_path,
                "image_description": description,
                "image_vector": image_vector.tolist(),
                "text_vector": text_vector.tolist()
            })
            
            logger.info(f"处理完成: {image_path}")
        
        # 批量插入数据
        if entities:
            self.client.insert(
                collection_name=self.collection_name,
                data=entities
            )
            logger.info(f"成功插入 {len(entities)} 条数据")
            
            # 刷新数据确保立即可搜索
            self.client.flush(self.collection_name)
        else:
            logger.warning("没有有效数据可插入")
    
    def load_collection(self):
        """加载集合到内存"""
        self.client.load_collection(self.collection_name)
        logger.info("集合已加载到内存")
    
    def search_by_text(self, query_text, top_k=5):
        """
        使用文本搜索相似图像
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            list: 搜索结果
        """
        # 将查询文本编码为向量
        query_vector = self.encode_text(query_text)
        if query_vector is None:
            return []
        
        # 执行向量搜索 - 用文本向量搜索图像向量
        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector.tolist()],
            anns_field="image_vector",  # 搜索图像向量字段
            search_params=search_params,
            limit=top_k,
            output_fields=["image_path", "image_description"]
        )
        
        return results[0] if results else []
    
    def search_by_image(self, query_image_path, top_k=5):
        """
        使用图像搜索相似文本/图像
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回结果数量
            
        Returns:
            list: 搜索结果
        """
        # 将查询图像编码为向量
        query_vector = self.encode_image(query_image_path)
        if query_vector is None:
            return []
        
        # 执行向量搜索
        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector.tolist()],
            anns_field="image_vector",  # 搜索图像向量字段
            search_params=search_params,
            limit=top_k,
            output_fields=["image_path", "image_description"]
        )
        
        return results[0] if results else []


def main():
    """主函数演示"""
    # 初始化搜索系统
    search_system = MultimodalSearchSystem()
    
    # 创建集合和索引
    search_system.create_collection()
    search_system.create_indexes()
    
    # 准备示例数据 - 可以使用本地图片或网络图片
    sample_data = [
        {
            "image_path": "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=400",
            "description": "一只可爱的橙色小猫坐在阳光下"
        },
        {
            "image_path": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400", 
            "description": "一只金毛犬在草地上奔跑"
        },
        {
            "image_path": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
            "description": "美丽的山脉和湖泊风景"
        },
        {
            "image_path": "https://images.unsplash.com/photo-1542831371-29b0f74f9713?w=400",
            "description": "现代化的城市建筑和天际线"
        },
        {
            "image_path": "https://images.unsplash.com/photo-1476224203421-9ac39bcb3327?w=400",
            "description": "五颜六色的花朵在花园中盛开"
        }
    ]
    
    # 插入数据
    logger.info("开始插入示例数据...")
    search_system.insert_data(sample_data)
    
    # 加载集合
    search_system.load_collection()
    
    # 演示文本搜索图像
    print("\n" + "="*50)
    print("📝 文本搜索图像示例")
    print("="*50)
    
    text_queries = [
        "可爱的小动物",
        "自然风景", 
        "城市建筑",
        "colorful flowers"
    ]
    
    for query in text_queries:
        print(f"\n🔍 查询: '{query}'")
        results = search_system.search_by_text(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. 相似度: {1-result.distance:.3f}")
            print(f"     描述: {result.entity.get('image_description')}")
            print(f"     图片: {result.entity.get('image_path')}")
    
    # 演示图像搜索
    print("\n" + "="*50)  
    print("🖼️ 图像搜索示例")
    print("="*50)
    
    # 使用第一张图片作为查询
    query_image = sample_data[0]["image_path"]
    print(f"\n🔍 使用图像查询: {query_image}")
    
    results = search_system.search_by_image(query_image, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. 相似度: {1-result.distance:.3f}")
        print(f"     描述: {result.entity.get('image_description')}")
        print(f"     图片: {result.entity.get('image_path')}")

if __name__ == "__main__":
    main() 