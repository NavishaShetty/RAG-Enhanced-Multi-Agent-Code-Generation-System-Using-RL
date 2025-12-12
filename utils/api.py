"""
OpenRouter API Client for LLM calls.
"""

import os
import yaml
import requests
from typing import Optional
from pathlib import Path


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class OpenRouterClient:
    """Client for making API calls to OpenRouter."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.
            model: Model to use. If not provided, uses config.yaml default.
        """
        config = load_config()

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = config["api"]["base_url"]
        self.model = model or config["api"]["model"]
        self.max_tokens = config["api"]["max_tokens"]
        self.temperature = config["api"]["temperature"]

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Make a completion call to the LLM.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt for context.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The LLM's response text.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/code-gen-rl",
            "X-Title": "Code Generation RL Agent"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(
                f"API call failed with status {response.status_code}: {response.text}"
            )

        result = response.json()

        if "choices" not in result or not result["choices"]:
            # API returned unexpected structure (e.g., rate limit, error)
            return ""

        content = result["choices"][0]["message"]["content"]
        return content or ""

    def test_connection(self) -> bool:
        """
        Test if the API connection is working.

        Returns:
            True if connection successful, raises exception otherwise.
        """
        try:
            response = self.call("Say 'Hello, connection test successful!'")
            return len(response) > 0
        except Exception as e:
            raise Exception(f"Connection test failed: {e}")


if __name__ == "__main__":
    # Quick test
    client = OpenRouterClient()
    print("Testing API connection...")
    try:
        client.test_connection()
        print("✓ Connection successful!")

        response = client.call("What is 2+2? Answer with just the number.")
        print(f"Test response: {response}")
    except Exception as e:
        print(f"✗ Error: {e}")
