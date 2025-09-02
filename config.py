import os
import pathlib
from typing import Optional


class Config:
    """Configuration class for adaptive paths and settings."""
    
    def __init__(self):
        self.project_root = self._get_project_root()
        self.arctic_db_path = self._get_arctic_db_path()
    
    def _get_project_root(self) -> pathlib.Path:
        """Get the project root directory."""
        current_file = pathlib.Path(__file__).resolve()
        return current_file.parent
    
    def _get_arctic_db_path(self) -> str:
        """Get the ArcticDB path, checking environment variable first."""
        # Check for environment variable override
        env_path = os.environ.get('CRYPTO_RESEARCH_DB_PATH')
        if env_path:
            db_path = pathlib.Path(env_path).resolve()
        else:
            # Default to project_root/arctic_store
            db_path = self.project_root / 'arctic_store'
        
        # Ensure directory exists
        db_path.mkdir(parents=True, exist_ok=True)
        
        return str(db_path)
    
    def get_arctic_uri(self) -> str:
        """Get the complete ArcticDB LMDB URI."""
        return f"lmdb://{self.arctic_db_path}"
    
    def get_data_path(self, relative_path: str = "") -> str:
        """Get path within the data directory."""
        data_path = self.project_root / 'data' / relative_path
        data_path.parent.mkdir(parents=True, exist_ok=True)
        return str(data_path)


# Global config instance
config = Config()

# Legacy compatibility - provide DB_PATH for existing notebooks
DB_PATH = config.arctic_db_path
ARCTIC_URI = config.get_arctic_uri()