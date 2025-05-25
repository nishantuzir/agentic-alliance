import logging
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """
    Set up and configure a logger with the specified name.
    
    Args:
        name (str): The name of the logger, typically __name__ of the calling module
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'agentic_alliance.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name) 