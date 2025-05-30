import asyncio
from ollama import AsyncClient

async def test_ollama():
    try:
        client = AsyncClient(host='http://localhost:11434')
        models = await client.list()
        print("Available models:")
        for model in models.get('models', []):
            print(f"- {model.get('name')} (modified: {model.get('modified_at')})")
            
        # Test the deepseek-r1:1.5b model
        print("\nTesting deepseek-r1:1.5b model...")
        response = await client.generate(
            model='deepseek-r1:1.5b',
            prompt='Hello, how are you?',
            stream=False
        )
        print("\nResponse:", response.get('response', 'No response'))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
