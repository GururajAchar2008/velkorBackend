"""
utils/retry.py

Reusable retry helper with exponential backoff.
"""

import time
import random
from functools import wraps
from typing import Callable, Tuple


DEFAULT_RETRY_STATUS = (408, 429, 500, 502, 503, 504)


def retry(
    max_retries: int = 2,
    base_delay: float = 1.0,
    retry_status: Tuple[int, ...] = DEFAULT_RETRY_STATUS,
):
    """
    Retry decorator.

    The wrapped function should either:
      - return an object with a 'status_code' attribute
      - or raise an exception.
    """

    def decorator(func: Callable):

        @wraps(func)
        def wrapper(*args, **kwargs):

            last_response = None

            for attempt in range(max_retries + 1):

                try:

                    response = func(*args, **kwargs)

                    last_response = response

                    status = getattr(response, "status_code", 200)

                    if status not in retry_status:
                        return response

                except Exception:

                    if attempt == max_retries:
                        raise

                if attempt < max_retries:

                    delay = (
                        base_delay
                        * (2 ** attempt)
                        + random.uniform(0, 0.5)
                    )

                    time.sleep(delay)

            return last_response

        return wrapper

    return decorator


def retry_request(func: Callable, *args, **kwargs):
    """
    Convenience helper.

    Example:

        result = retry_request(api.generate, messages)
    """

    wrapped = retry()(func)

    return wrapped(*args, **kwargs)
