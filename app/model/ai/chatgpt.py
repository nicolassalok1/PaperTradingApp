import os
from typing import Tuple

# Make the OpenAI dependency optional so the rest of the app can run without it.
try:  # pragma: no cover - import guard
    import openai  # type: ignore
except Exception:  # noqa: BLE001
    openai = None  # type: ignore

from app.utils.secrets import get_secret


def _get_openai_client() -> Tuple[object, str]:
    """
    Returns either an OpenAI client/module or an error message.
    Supports both the legacy SDK (<1.x) and the newer client API (>=1.x).
    """
    if openai is None:
        return None, "OpenAI SDK not installed"

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY not configured"

    try:
        # New SDK style (>=1.x)
        if hasattr(openai, "OpenAI"):
            return openai.OpenAI(api_key=api_key), ""

        # Legacy SDK style (<1.x)
        openai.api_key = api_key
        return openai, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"failed to initialize OpenAI client: {exc}"


def chatgpt_response(prompt: str) -> str:
    """
    Minimal ChatGPT wrapper used by the Dashboard assistant.
    Returns a friendly error message if the backend call fails.
    """
    client, err = _get_openai_client()
    if err:
        return f"[ChatGPT unavailable: {err}]"

    try:
        # New SDK style
        if hasattr(client, "chat"):
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return completion.choices[0].message.content

        # Legacy SDK style
        completion = client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message["content"]
    except Exception as exc:  # noqa: BLE001
        return f"[ChatGPT backend error: {exc}]"
