import json
import logging
import time

logger = logging.getLogger(__name__)


def log_action(event: str, metadata: dict):
    entry = {
        "timestamp": time.time(),
        "event": event,
        "metadata": metadata,
    }
    logger.info("options_activity %s", json.dumps(entry))


__all__ = ["log_action"]
