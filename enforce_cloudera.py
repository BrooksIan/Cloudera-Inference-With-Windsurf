#!/usr/bin/env python3
"""
Script to enforce Cloudera-only models for Cascade/Windsurf.
This should be imported early in the application startup to ensure
only Cloudera-hosted models are used.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def enforce_cloudera_models():
    """Monkey patch the config to ensure only Cloudera models are used."""
    try:
        # Import the Cloudera configuration
        from scratch.cloudera_config import ClouderaConfig, ClouderaLLMConfig
        
        # Monkey patch the config module
        import windsurf_agent.config as config_module
        
        # Replace the original classes with Cloudera-enforced versions
        config_module.Config = ClouderaConfig
        config_module.LLMConfig = ClouderaLLMConfig
        
        print("✅ Enforced Cloudera-hosted models only policy")
        print("   - Only Cloudera endpoints will be allowed")
        print("   - Model validation enabled")
        print("   - Non-Cloudera services will be blocked")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Cloudera config: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to enforce Cloudera models: {e}")
        return False

def verify_cloudera_enforcement():
    """Verify that Cloudera enforcement is active."""
    try:
        import windsurf_agent.config as config_module
        
        # Check if the classes have been patched
        config_class = config_module.Config
        llm_config_class = config_module.LLMConfig
        
        # Check class names to verify patching
        config_name = config_class.__name__
        llm_config_name = llm_config_class.__name__
        
        if config_name == "ClouderaConfig" and llm_config_name == "ClouderaLLMConfig":
            print("✅ Cloudera enforcement verified - both classes patched")
            return True
        else:
            print(f"⚠️  Partial enforcement detected:")
            print(f"   Config class: {config_name}")
            print(f"   LLM Config class: {llm_config_name}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to verify enforcement: {e}")
        return False

def test_cloudera_validation():
    """Test the Cloudera validation with current configuration."""
    try:
        from scratch.cloudera_config import ClouderaLLMConfig
        
        # Test with current environment
        base_url = os.getenv("WINDSURF_LLM_BASE_URL", "")
        api_key = os.getenv("WINDSURF_LLM_API_KEY", "")
        model = os.getenv("WINDSURF_LLM_MODEL", "")
        
        print(f"🔍 Testing Cloudera validation...")
        print(f"   Base URL: {base_url}")
        print(f"   Model: {model}")
        
        if not base_url:
            print("❌ No base URL configured")
            return False
            
        try:
            config = ClouderaLLMConfig(
                base_url=base_url,
                api_key=api_key,
                model=model
            )
            print("✅ Cloudera validation passed")
            return True
        except ValueError as e:
            print(f"❌ Cloudera validation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Cloudera Model Enforcement for Cascade")
    print("=" * 50)
    
    # Test current configuration
    print("\n📋 Step 1: Testing current configuration...")
    test_passed = test_cloudera_validation()
    
    # Enforce Cloudera models
    print("\n📋 Step 2: Enforcing Cloudera models...")
    enforcement_passed = enforce_cloudera_models()
    
    # Verify enforcement
    print("\n📋 Step 3: Verifying enforcement...")
    verification_passed = verify_cloudera_enforcement()
    
    # Summary
    print(f"\n📊 Results:")
    print(f"Configuration Test: {'✅ PASS' if test_passed else '❌ FAIL'}")
    print(f"Enforcement Applied: {'✅ PASS' if enforcement_passed else '❌ FAIL'}")
    print(f"Verification: {'✅ PASS' if verification_passed else '❌ FAIL'}")
    
    if all([test_passed, enforcement_passed, verification_passed]):
        print(f"\n🎉 Cloudera enforcement successfully applied!")
        print(f"   Cascade will now only use Cloudera-hosted models.")
    else:
        print(f"\n⚠️  Some issues detected. Please check the configuration.")
