"""
Base extractor class with rate limiting and error handling
"""

import time
from typing import Optional, Any
from abc import ABC, abstractmethod

from config.settings import ETLConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseExtractor(ABC):
    """Base class for all data extractors"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
    
    def rate_limit(self):
        """Implement rate limiting to avoid overwhelming NBA API"""
        self.request_count += 1
        time.sleep(ETLConfig.REQUEST_DELAY)
        
        if self.request_count % ETLConfig.BATCH_SIZE == 0:
            logger.info(
                f"Processed {self.request_count} requests, "
                f"taking {ETLConfig.BATCH_DELAY}s pause..."
            )
            time.sleep(ETLConfig.BATCH_DELAY)
    
    def extract_with_retry(
        self,
        extract_func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Execute extraction with retry logic and exponential backoff
        
        Args:
            extract_func: Function to execute
            *args: Positional arguments for function
            max_retries: Maximum retry attempts (uses config default if None)
            **kwargs: Keyword arguments for function
            
        Returns:
            Result from extract_func or None if all retries failed
        """
        if max_retries is None:
            max_retries = ETLConfig.MAX_RETRIES
        
        for attempt in range(max_retries + 1):
            try:
                self.rate_limit()
                result = extract_func(*args, **kwargs)
                return result
                
            except Exception as e:
                self.error_count += 1
                error_msg = str(e)
                
                # Check if it's a timeout or rate limit error
                is_timeout = 'timed out' in error_msg.lower() or 'timeout' in error_msg.lower()
                is_rate_limit = 'rate' in error_msg.lower() or '429' in error_msg
                
                if attempt < max_retries:
                    # Exponential backoff for timeouts/rate limits
                    if is_timeout or is_rate_limit:
                        backoff_time = ETLConfig.RETRY_DELAY * (2 ** attempt)  # 5s, 10s, 20s, 40s
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Backing off for {backoff_time}s..."
                        )
                        time.sleep(backoff_time)
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {ETLConfig.RETRY_DELAY}s..."
                        )
                        time.sleep(ETLConfig.RETRY_DELAY)
                else:
                    logger.error(
                        f"All {max_retries + 1} attempts failed for {extract_func.__name__}: {e}"
                    )
                    return None
        
        return None
    
    @abstractmethod
    def extract(self, *args, **kwargs):
        """
        Main extraction method to be implemented by subclasses
        
        This method should contain the core extraction logic
        """
        pass
    
    def get_stats(self) -> dict:
        """
        Get extraction statistics
        
        Returns:
            Dictionary with request and error counts
        """
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'success_rate': (
                (self.request_count - self.error_count) / self.request_count * 100
                if self.request_count > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset extraction statistics"""
        self.request_count = 0
        self.error_count = 0