import time
from datetime import datetime, timedelta

from fastapi import HTTPException, Request
from utils.agentic_alliance_logger import setup_logger

logger = setup_logger(__name__)

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        logger.info(f"Initializing RateLimiter with {requests_per_minute} requests per minute")
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list] = {}
        
    def is_allowed(self, token: str) -> bool:
        logger.debug(f"Checking rate limit for token: {token[:8]}...")
        now = time.time()
        if token not in self.requests:
            logger.debug(f"First request for token: {token[:8]}...")
            self.requests[token] = []
        
        # Remove requests older than 1 minute
        self.requests[token] = [req_time for req_time in self.requests[token] 
                              if now - req_time < 60]
        
        if len(self.requests[token]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for token: {token[:8]}...")
            return False
            
        self.requests[token].append(now)
        logger.debug(f"Request allowed for token: {token[:8]}... ({len(self.requests[token])}/{self.requests_per_minute} requests)")
        return True

class TokenManager:
    def __init__(self, expiration_minutes: int = 60):
        logger.info(f"Initializing TokenManager with {expiration_minutes} minutes expiration")
        self.expiration_minutes = expiration_minutes
        self.tokens: dict[str, datetime] = {}
        
    def is_valid(self, token: str) -> bool:
        logger.debug(f"Validating token: {token[:8]}...")
        if token not in self.tokens:
            logger.warning(f"Token not found: {token[:8]}...")
            return False
        if datetime.now() > self.tokens[token]:
            logger.warning(f"Token expired: {token[:8]}...")
            del self.tokens[token]
            return False
        logger.debug(f"Token valid: {token[:8]}...")
        return True
        
    def add_token(self, token: str) -> None:
        logger.info(f"Adding new token: {token[:8]}...")
        self.tokens[token] = datetime.now() + timedelta(minutes=self.expiration_minutes)
        logger.debug(f"Token will expire at: {self.tokens[token]}")

# Initialize managers
logger.info("Initializing auth managers")
token_manager = TokenManager()
rate_limiter = RateLimiter()

async def verify_token(request: Request, token: str) -> None:
    logger.info(f"Verifying token: {token[:8]}...")
    if not token_manager.is_valid(token):
        logger.error(f"Invalid token: {token[:8]}...")
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    if not rate_limiter.is_allowed(token):
        logger.error(f"Rate limit exceeded for token: {token[:8]}...")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    logger.debug(f"Token verification successful: {token[:8]}...")
