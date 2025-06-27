#!/usr/bin/env python3

import os
from deepsearcher.configuration import Configuration

def test_docker_config():
    """Test if docker-config.yaml can be loaded correctly"""
    try:
        # Test docker config path
        docker_config_path = "docker-config.yaml"
        if os.path.exists(docker_config_path):
            print(f"Found docker-config.yaml at: {docker_config_path}")
            
            # Try to load the configuration
            config = Configuration(docker_config_path)
            print("✅ Configuration loaded successfully!")
            
            # Check if provide_settings exists
            if hasattr(config, 'provide_settings'):
                print("✅ provide_settings found!")
                print(f"Available features: {list(config.provide_settings.keys())}")
                
                # Check each feature
                for feature in config.provide_settings:
                    provider_info = config.get_provider_config(feature)
                    print(f"  {feature}: {provider_info['provider']}")
                    
            else:
                print("❌ provide_settings not found!")
                
        else:
            print(f"❌ docker-config.yaml not found at {docker_config_path}")
            
        # Test default config
        print("\nTesting default config...")
        default_config = Configuration()
        print("✅ Default configuration loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_docker_config() 