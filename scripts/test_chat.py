#!/usr/bin/env python3
"""
Test chat script for quick API testing
"""
import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiohttp


async def test_chat_api():
    """Test the chat API endpoint"""
    base_url = "http://localhost:8000"
    
    # Test data
    test_messages = [
        "Hello, I'm feeling a bit lonely today.",
        "Can you help me find some podcasts about history?",
        "I need to remember to take my blood pressure medication.",
        "What's in the news today?"
    ]
    
    print("🧪 Testing Elder Care Assistant API")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        print("1. Testing health endpoint...")
        try:
            async with session.get(f"{base_url}/api/v1/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    print(f"   ✅ Health check passed: {health_data['status']}")
                else:
                    print(f"   ❌ Health check failed: {resp.status}")
                    return
        except Exception as e:
            print(f"   ❌ Cannot connect to server: {e}")
            print("   💡 Make sure to run: python -m src.api.app")
            return
        
        # Test chat endpoint
        print("\n2. Testing chat endpoint...")
        for i, message in enumerate(test_messages, 1):
            print(f"\n   Test {i}: '{message}'")
            
            payload = {
                "message": message,
                "user_id": "test_user_123",
                "platform": "api",
                "elder_mode": True
            }
            
            try:
                async with session.post(
                    f"{base_url}/api/v1/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        response_data = await resp.json()
                        print(f"   ✅ Response: {response_data['response'][:100]}...")
                        
                        metadata = response_data.get('metadata', {})
                        if metadata.get('tools_used'):
                            print(f"   🔧 Tools used: {metadata['tools_used']}")
                    else:
                        error_text = await resp.text()
                        print(f"   ❌ Error {resp.status}: {error_text}")
                        
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
        
        # Test tools endpoint
        print("\n3. Testing tools endpoint...")
        try:
            async with session.get(f"{base_url}/api/v1/tools") as resp:
                if resp.status == 200:
                    tools_data = await resp.json()
                    print(f"   ✅ Found {len(tools_data['tools'])} tools:")
                    for tool in tools_data['tools']:
                        print(f"      • {tool['name']}: {tool['description']}")
                else:
                    print(f"   ❌ Tools endpoint failed: {resp.status}")
        except Exception as e:
            print(f"   ❌ Tools request failed: {e}")
    
    print("\n🎉 API testing complete!")


if __name__ == "__main__":
    asyncio.run(test_chat_api())