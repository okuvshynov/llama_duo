import requests
import json

# Server configuration
SERVER_URL = "http://localhost:5555" 

def send_query(offset, text, complete):
    """Send a query to the server."""
    endpoint = f"{SERVER_URL}/query"
    payload = {
        "offset": offset,
        "text": text,
        "complete": complete
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        with requests.post(endpoint, data=json.dumps(payload), headers=headers, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk.decode('utf-8')
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        yield None

def main():
    # Test cases
    test_cases = [
        (0, "Hello", False),
        (5, " World", False),
        (0, "Greetings", True),
        (10, "!", True)
    ]
    
    for offset, text, ready in test_cases:
        print(f"\nSending query: offset={offset}, text='{text}', ready={ready}")
        for chunk in send_query(offset, text, ready):
            if chunk is None:
                print("Failed to get response from server.")
                break
            print(f"Received chunk: {chunk}")

if __name__ == "__main__":
    main()
