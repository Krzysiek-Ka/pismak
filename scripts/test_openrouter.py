from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._shared.serwer import OPENROUTER_KEY_PLACEHOLDER


def main() -> None:
    """Send one short manual smoke-test request without printing the API key."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("OPENROUTER_KEY", "").strip()
    if not api_key or api_key == OPENROUTER_KEY_PLACEHOLDER:
        raise SystemExit("Brak OPENROUTER_KEY. Ustaw klucz w pliku .env.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=60.0,
        max_retries=0,
    )
    try:
        completion = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": "Odpowiedz jednym słowem: gotowe."}],
            max_tokens=16,
        )
    except Exception as error:
        raise SystemExit(f"Zapytanie nie powiodło się ({type(error).__name__}).") from None

    print(f"Model: {completion.model or 'nie podano'}")
    print(completion.choices[0].message.content or "")


if __name__ == "__main__":
    main()
