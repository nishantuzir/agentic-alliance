
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import HTTPException, Request
import time

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
        
    def is_allowed(self, token: str) -> bool:
        now = time.time()
        if token not in self.requests:
            self.requests[token] = []
        
        # Remove requests older than 1 minute
        self.requests[token] = [req_time for req_time in self.requests[token] 
                              if now - req_time < 60]
        
        if len(self.requests[token]) >= self.requests_per_minute:
            return False
            
        self.requests[token].append(now)
        return True

class TokenManager:
    def __init__(self, expiration_minutes: int = 60):
        self.expiration_minutes = expiration_minutes
        self.tokens: Dict[str, datetime] = {}
        
    def is_valid(self, token: str) -> bool:
        if token not in self.tokens:
            return False
        if datetime.now() > self.tokens[token]:
            del self.tokens[token]
            return False
        return True
        
    def add_token(self, token: str) -> None:
        self.tokens[token] = datetime.now() + timedelta(minutes=self.expiration_minutes)

# Initialize managers
token_manager = TokenManager()
rate_limiter = RateLimiter()

async def verify_token(request: Request, token: str) -> None:
    if not token_manager.is_valid(token):
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    if not rate_limiter.is_allowed(token):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
