#!/usr/bin/env python3
"""
Quick smoke test for the /ws/aerivon Command Center endpoint
"""
import asyncio
import websockets
import json

async def test_aerivon():
    uri = "ws://127.0.0.1:8081/ws/aerivon"
    print(f"🔗 Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("✅ WebSocket connected!")
        
        # Send start message
        start_msg = {
            "type": "start",
            "user_id": "test_user_123",
            "memory_scope": "aerivon_global"
        }
        await websocket.send(json.dumps(start_msg))
        print(f"📤 Sent: {start_msg}")
        
        # Send a story request to test intent detection
        story_request = {
            "type": "text",
            "text": "Create a short fantasy story about a brave knight"
        }
        await websocket.send(json.dumps(story_request))
        print(f"📤 Sent: {story_request}")
        
        # Receive responses
        print("\n📥 Receiving responses:")
        message_count = 0
        max_messages = 30  # Limit to avoid infinite loop
        
        while message_count < max_messages:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=60.0)  # 60s timeout for story generation
                data = json.loads(response)
                message_count += 1
                
                msg_type = data.get("type")
                
                if msg_type == "status":
                    print(f"  ⚡ STATUS: {data.get('state')}")
                elif msg_type == "intent":
                    print(f"  🎯 INTENT: {data.get('intent')} (confidence: {data.get('confidence')})")
                    print(f"     Reason: {data.get('reason')}")
                elif msg_type == "thinking":
                    print(f"  💭 THINKING: {data.get('text')[:80]}...")
                elif msg_type == "text":
                    text = data.get('text', '')
                    print(f"  💬 TEXT: {text[:100]}{'...' if len(text) > 100 else ''}")
                elif msg_type == "image":
                    print(f"  🖼️  IMAGE: Received image #{data.get('index', 0)} ({data.get('mime_type')})")
                elif msg_type == "done":
                    print(f"  ✅ DONE")
                    break
                elif msg_type == "error":
                    print(f"  ❌ ERROR: {data.get('error')}")
                    break
                else:
                    print(f"  ❓ Unknown type: {msg_type}")
                    
            except asyncio.TimeoutError:
                print("⏱️  Timeout waiting for response")
                break
            except Exception as e:
                print(f"❌ Error receiving message: {e}")
                break
        
        print(f"\n✅ Test completed! Received {message_count} messages")

if __name__ == "__main__":
    asyncio.run(test_aerivon())
