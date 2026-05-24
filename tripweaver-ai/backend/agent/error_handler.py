"""
Centralized Error Handler
--------------------------
Provides:
  - ToolError — typed exception for tool failures
  - with_retry — decorator for automatic retry with exponential back-off
  - safe_tool_call — wrapper that catches all exceptions and returns a
    user-friendly error string instead of crashing the agent
"""

from __future__ import annotations

import time
import functools
import traceback
from typing import Callable, TypeVar, Any

F = TypeVar("F", bound=Callable[..., Any])


class ToolError(Exception):
    """Raised when a tool encounters a recoverable error."""
    pass


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator: retry a function up to `max_attempts` times on failure.

    Args:
        max_attempts: Total number of attempts (including the first).
        delay:        Initial wait in seconds between attempts.
        backoff:      Multiplier applied to delay after each failure.
        exceptions:   Tuple of exception types to catch and retry on.

    Example:
        @with_retry(max_attempts=3, delay=1.0)
        def call_api(city: str) -> str: ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    # On final attempt, fall through and raise

            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]
    return decorator


def safe_tool_call(func: Callable, *args, tool_name: str = "", **kwargs) -> str:
    """
    Call a tool function and catch all exceptions.
    Returns a user-friendly error string instead of propagating the exception.

    Usage:
        result = safe_tool_call(get_weather, "Goa", tool_name="WeatherTool")
    """
    try:
        return func(*args, **kwargs)
    except ToolError as exc:
        return f"⚠️ {tool_name or func.__name__} error: {exc}"
    except requests_timeout_error() as exc:
        return f"⏱️ {tool_name or func.__name__} timed out. Please try again."
    except Exception as exc:
        # Log full traceback for debugging, return clean message to user
        tb = traceback.format_exc()
        print(f"[ERROR] {tool_name or func.__name__} failed:\n{tb}")
        return (
            f"❌ {tool_name or func.__name__} encountered an unexpected error. "
            "Please try again or rephrase your question."
        )


def requests_timeout_error():
    """Return requests.Timeout if available, else a dummy that never matches."""
    try:
        import requests
        return requests.Timeout
    except ImportError:
        return type("_NeverMatch", (Exception,), {})


# ── Tool result validators ────────────────────────────────────────────────────

def validate_city(city: str) -> str:
    """
    Validate and normalise a city name input.
    Raises ToolError if the input is empty or clearly invalid.
    """
    city = city.strip()
    if not city:
        raise ToolError("City name cannot be empty.")
    if len(city) > 100:
        raise ToolError(f"City name too long: '{city[:50]}…'")
    # Strip accidental extra words like "city of Goa" → "Goa"
    for prefix in ("city of ", "town of ", "state of ", "district of "):
        if city.lower().startswith(prefix):
            city = city[len(prefix):]
    return city.title()


def validate_budget_input(input_text: str) -> tuple[float, int]:
    """
    Parse and validate 'AMOUNT,DAYS' budget input.
    Returns (amount, days) tuple.
    Raises ToolError with a clear message on bad input.
    """
    parts = [p.strip() for p in input_text.split(",")]
    if len(parts) != 2:
        raise ToolError(
            "Invalid format. Expected 'AMOUNT,DAYS' e.g. '15000,3'. "
            f"Got: '{input_text}'"
        )
    try:
        amount = float(parts[0].replace("₹", "").replace("rs", "").replace("inr", "").strip())
        days = int(parts[1])
    except ValueError:
        raise ToolError(
            f"Could not parse numbers from '{input_text}'. "
            "Use digits only, e.g. '15000,3'."
        )
    if amount <= 0:
        raise ToolError("Budget amount must be greater than ₹0.")
    if days <= 0:
        raise ToolError("Number of days must be at least 1.")
    return amount, days
