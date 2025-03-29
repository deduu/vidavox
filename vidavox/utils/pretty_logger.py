import json

def pretty_json_log(logger, data, message=""):
    """Logs a dictionary or JSON-serializable object with pretty formatting."""
    try:
        json_str = json.dumps(data, indent=4)
        logger.info(f"{message} {json_str}")
    except TypeError:
        logger.error("Data is not JSON serializable.")
    except Exception as e:
        logger.error(f"Error during JSON formatting: {e}")