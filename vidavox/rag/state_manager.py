
# State management module (state_manager.py)
import os
import logging
import pickle
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateManager:
    """Handles persistence of Retrieval engine state."""
    
    @staticmethod
    def save_state(path: str, state_data: Dict[str, Any]) -> bool:
        """Save engine state to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(f"{path}_state.pkl", 'wb') as f:
                pickle.dump(state_data, f)
            logger.info(f"State successfully saved to {path}_state.pkl")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    @staticmethod
    def load_state(path: str) -> Optional[Dict[str, Any]]:
        """Load engine state from disk."""
        if os.path.exists(f"{path}_state.pkl"):
            try:
                with open(f"{path}_state.pkl", 'rb') as f:
                    state_data = pickle.load(f)
                logger.info(f"State successfully loaded from {path}_state.pkl")
                return state_data
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return None
        else:
            logger.info(f"No state file found at {path}_state.pkl")
            return None

