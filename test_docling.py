#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing Docling functionality...")

try:
    from deepsearcher.configuration import Configuration, init_config
    print("✓ DeepSearcher configuration imported successfully")
    
    config = Configuration()
    
    # Configure providers
    config.set_provider_config("llm", "SiliconFlow", {"model": "deepseek-ai/DeepSeek-V3"})
    config.set_provider_config("embedding", "SiliconflowEmbedding", {"model": "BAAI/bge-m3"})
    config.set_provider_config("vector_db", "Milvus", {
        "default_collection": "deepsearcher",
        "uri": "http://localhost:19530",
        "token": "root:Milvus",
        "db": "default"
    })
    config.set_provider_config("file_loader", "DoclingLoader", {})
    
    print("✓ Configuration set successfully")
    
    init_config(config)
    print("✓ Configuration initialized successfully")
    
    # Test file loading
    from deepsearcher.offline_loading import load_from_local_files
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "examples/data/WhatisMilvus.pdf")
    
    if os.path.exists(test_file):
        print(f"✓ Test file found: {test_file}")
        
        print("Loading file with DoclingLoader...")
        load_from_local_files(
            paths_or_directory=test_file,
            collection_name="DoclingTest",
            collection_description="Test collection for Docling",
            force_new_collection=True
        )
        print("✓ File loaded successfully!")
        
        # Test query
        from deepsearcher.online_query import query
        
        print("Testing query...")
        result, _, tokens = query("What is Milvus?", max_iter=1)
        print(f"✓ Query completed successfully! Tokens used: {tokens}")
        print(f"Result: {result[:200]}...")
        
    else:
        print(f"✗ Test file not found: {test_file}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!") 