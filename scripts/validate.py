#!/usr/bin/env python3
"""
Quick validation script to test that all components work together
"""
import sys
import os
sys.path.append('.')

def main():
    print("🧪 Validating Elder Care AI Assistant...")
    
    try:
        # Test core imports
        from src.agent.graph import ElderlyAssistantAgent
        from src.memory.manager import MemoryManager
        from src.settings import settings
        print("✅ Core imports successful")
        
        # Check settings
        has_openai_key = bool(settings.openai_api_key)
        print(f"✅ Settings loaded (OpenAI key configured: {has_openai_key})")
        
        # Test memory manager
        manager = MemoryManager()
        print("✅ Memory manager initialized")
        
        # Test agent creation
        agent = ElderlyAssistantAgent()
        print("✅ Agent created successfully")
        
        # Check if database directory exists
        db_path = os.path.dirname(settings.memory_db_path)
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
            print(f"✅ Created database directory: {db_path}")
        
        print("\n🎉 All components validated successfully!")
        print("\n📋 Next steps:")
        
        if not has_openai_key:
            print("1. ⚠️  Add your OpenAI API key to .env file")
            print("   Copy .env.example to .env and add: OPENAI_API_KEY=your_key_here")
        else:
            print("1. ✅ OpenAI API key is configured")
            
        print("2. 🚀 Test locally: python scripts/run_local.py")
        print("3. 🌐 Start API server: python -m src.api.app")
        print("4. 📖 View API docs at: http://localhost:8000/docs")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Try: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)