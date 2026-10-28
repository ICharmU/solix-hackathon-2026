"""
Test connection script for OpenAI gpt-4o-mini.
Verifies API credentials, endpoint reachability, model availability, and response generation.
"""

import os
import sys
import time
from dotenv import load_dotenv

def test_gpt_4o_mini():
    print("=" * 60)
    print("      Testing Connection to OpenAI (Model: gpt-4o-mini)      ")
    print("=" * 60)

    # Load environment variables (.env file or system env)
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "your_" in api_key.lower() or api_key.strip() == "":
        print("\n[ERROR] Valid OPENAI_API_KEY is not set.")
        print("\nTo configure your API key, update the `.env` file in the project root with:")
        print("  OPENAI_API_KEY=sk-proj-...")
        print("Or export it in PowerShell:")
        print("  $env:OPENAI_API_KEY='sk-proj-...'")
        print("=" * 60)
        sys.exit(1)

    # Mask key for privacy
    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"\n[INFO] Found OPENAI_API_KEY: {masked_key}")
    print("[INFO] Initializing OpenAI client...")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"\n[ERROR] Failed to instantiate OpenAI client: {e}")
        sys.exit(1)

    print("[INFO] Sending test request to 'gpt-4o-mini'...")
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a diagnostic assistant. Respond concisely to connection verification tests."
                },
                {
                    "role": "user",
                    "content": "Hello! Confirm that you are gpt-4o-mini and the API connection is active."
                }
            ],
            max_tokens=60,
            temperature=0.2,
        )
        latency = time.time() - start_time
        reply = response.choices[0].message.content.strip()
        usage = response.usage

        print("\n" + "-" * 60)
        print("[SUCCESS] Successfully connected to OpenAI API!")
        print(f"Model ID:            {response.model}")
        print(f"Response Latency:    {latency:.2f} seconds")
        if usage:
            print(f"Prompt Tokens:       {usage.prompt_tokens}")
            print(f"Completion Tokens:   {usage.completion_tokens}")
            print(f"Total Tokens:        {usage.total_tokens}")
        print("-" * 60)
        print(f"Model Response:\n\"{reply}\"")
        print("=" * 60)
        print("\nConnection test passed. Ready for pipeline implementation.")
        return True

    except Exception as e:
        latency = time.time() - start_time
        print("\n" + "!" * 60)
        print(f"[FAILED] Connection test failed after {latency:.2f} seconds.")
        print(f"Error Type:    {type(e).__name__}")
        print(f"Error Message: {e}")
        print("!" * 60)
        sys.exit(1)

if __name__ == "__main__":
    test_gpt_4o_mini()
