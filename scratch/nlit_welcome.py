#!/usr/bin/env python3
"""
NLIT 2026 Conference Welcome Script
Uses Cloudera-hosted models only to generate a welcome message.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

def generate_nlit_welcome():
    """Generate a welcome message for NLIT 2026 using Cloudera models."""
    
    # Import Cloudera-enforced configuration
    import windsurf_agent
    from windsurf_agent.config import Config
    from windsurf_agent.ClouderaLLMClient import ClouderaLLMClient
    
    print("🚀 Generating NLIT 2026 Welcome Message")
    print("=" * 50)
    print("🔒 Using Cloudera-hosted models only")
    print("=" * 50)
    
    try:
        # Load configuration (automatically enforced to use Cloudera)
        config = Config.from_env()
        
        print(f"📍 LLM Endpoint: {config.llm.base_url}")
        print(f"🤖 Model: {config.llm.model}")
        print(f"🌡️  Temperature: {config.llm.temperature}")
        
        # Create Cloudera LLM client (uses environment variables automatically)
        client = ClouderaLLMClient()
        
        # Generate welcome message
        prompt = """Create a warm, professional welcome message for the NLIT 2026 conference. 

The message should:
- Welcome attendees to NLIT 2026
- Mention the importance of natural language processing and information technology
- Be inspiring and forward-looking
- Be approximately 150-200 words
- Include a call to action for networking and learning
- Have a professional yet friendly tone

Please write the welcome message:"""

        print("\n📝 Generating welcome message...")
        
        # Collect the streamed response
        welcome_chunks = []
        for chunk in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            stream=False  # Disable streaming for this use case
        ):
            welcome_chunks.append(chunk)
        
        welcome_message = ''.join(welcome_chunks)
        
        print("\n" + "=" * 60)
        print("🎉 NLIT 2026 WELCOME MESSAGE")
        print("=" * 60)
        print(welcome_message)
        print("=" * 60)
        
        return welcome_message
        
    except Exception as e:
        print(f"❌ Error generating welcome message: {e}")
        return None

def save_welcome_message(message, filename="nlit_2026_welcome.txt"):
    """Save the welcome message to a file."""
    if message:
        try:
            output_path = Path(__file__).parent / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("NLIT 2026 Conference Welcome Message\n")
                print("=" * 50)
                print(f"Generated using Cloudera-hosted models\n")
                print(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                print("=" * 50)
                print("\n")
                f.write(message)
            
            print(f"\n💾 Welcome message saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving message: {e}")

def verify_cloudera_enforcement():
    """Verify that we're using Cloudera models only."""
    import windsurf_agent.config as config_module
    
    config_class = config_module.Config.__name__
    llm_config_class = config_module.LLMConfig.__name__
    
    if config_class == "ClouderaConfig" and llm_config_class == "ClouderaLLMConfig":
        print("✅ Confirmed: Using Cloudera-enforced configuration")
        return True
    else:
        print(f"⚠️  Warning: Not using Cloudera enforcement")
        print(f"   Config: {config_class}")
        print(f"   LLM Config: {llm_config_class}")
        return False

if __name__ == "__main__":
    print("🎪 NLIT 2026 Conference Welcome Generator")
    print("🔒 Cloudera Models Only Enforcement")
    print("=" * 60)
    
    # Verify Cloudera enforcement
    if not verify_cloudera_enforcement():
        print("❌ Cloudera enforcement not active. Exiting.")
        sys.exit(1)
    
    # Generate welcome message
    welcome_message = generate_nlit_welcome()
    
    # Save to file
    save_welcome_message(welcome_message)
    
    print("\n🎊 NLIT 2026 welcome message generation complete!")
    print("🤖 Powered by Cloudera-hosted AI models")
