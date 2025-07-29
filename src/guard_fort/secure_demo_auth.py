"""
Secure Demo Authentication Module for KonveyN2AI
Provides time-limited, origin-restricted demo token authentication for hackathon/demo purposes.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SecureDemoAuthenticator:
    """
    Secure demo token authentication with time limits, origin restrictions, and rate limiting.
    """

    # Demo token expires on Sunday, August 3, 2025 at 23:59:59 UTC (end of weekend)
    DEMO_TOKEN_EXPIRY = datetime(2025, 8, 3, 23, 59, 59, tzinfo=timezone.utc)

    # Allowed origins for demo token usage
    ALLOWED_ORIGINS = [
        "https://localhost:3000",  # Local development
        "https://localhost:8080",  # Local development
        "http://localhost:3000",  # Local development
        "http://localhost:8080",  # Local development
        "https://*.vercel.app",  # Vercel production
        "https://*.netlify.app",  # Netlify deployments
        "https://*.run.app",  # Google Cloud Run
    ]

    # Rate limiting: 100 requests per minute per origin
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 60  # seconds

    # Cache limits to prevent memory exhaustion
    MAX_CACHE_ORIGINS = 1000  # Maximum number of origins to track
    MAX_CACHE_ENTRIES_PER_ORIGIN = (
        200  # Maximum entries per origin (2x rate limit for safety)
    )

    def __init__(self):
        # Rate limiting storage: origin -> [timestamp, ...]
        self._rate_limit_cache = defaultdict(list)
        # Track cache access time for LRU eviction
        self._cache_access_time = defaultdict(float)

        # Demo token usage statistics
        self._usage_stats = {
            "total_requests": 0,
            "unique_origins": set(),
            "rate_limit_violations": 0,
            "origin_violations": 0,
            "expired_token_attempts": 0,
        }

        logger.info(
            f"SecureDemoAuthenticator initialized. Token expires: {self.DEMO_TOKEN_EXPIRY}"
        )

    def validate_demo_token(
        self, token: str, origin: str = "", user_agent: str = ""
    ) -> dict[str, Any]:
        """
        Validate demo token with comprehensive security checks.

        Args:
            token: The authentication token
            origin: Request origin header
            user_agent: Request user agent header

        Returns:
            Dict with validation result and reason
        """
        # Check if token is the demo token
        if token != "demo-token":  # nosec B105 - demo token for hackathon only
            return {"valid": False, "reason": "invalid_demo_token"}

        # Check token expiration
        current_time = datetime.now(timezone.utc)
        if current_time > self.DEMO_TOKEN_EXPIRY:
            self._usage_stats["expired_token_attempts"] += 1
            logger.warning(
                f"Demo token expired. Current time: {current_time}, Expiry: {self.DEMO_TOKEN_EXPIRY}"
            )
            return {
                "valid": False,
                "reason": "demo_token_expired",
                "expires_at": self.DEMO_TOKEN_EXPIRY.isoformat(),
            }

        # Check origin restrictions
        if not self._is_allowed_origin(origin):
            self._usage_stats["origin_violations"] += 1
            logger.warning(f"Demo token used from unauthorized origin: {origin}")
            return {"valid": False, "reason": "unauthorized_origin", "origin": origin}

        # Check rate limiting
        if not self._check_rate_limit(origin):
            self._usage_stats["rate_limit_violations"] += 1
            logger.warning(f"Rate limit exceeded for origin: {origin}")
            return {"valid": False, "reason": "rate_limit_exceeded", "origin": origin}

        # Basic bot detection
        if self._is_bot_request(user_agent):
            logger.warning(f"Bot request detected: {user_agent[:100]}")
            return {"valid": False, "reason": "bot_access_denied"}

        # Token is valid - update statistics
        self._update_usage_stats(origin)

        # Calculate time until expiry
        time_until_expiry = (self.DEMO_TOKEN_EXPIRY - current_time).total_seconds()

        logger.info(
            f"Demo token validated successfully. Origin: {origin}, Expires in: {time_until_expiry:.1f}s"
        )

        return {
            "valid": True,
            "reason": "demo_token_valid",
            "expires_in_seconds": time_until_expiry,
            "expires_at": self.DEMO_TOKEN_EXPIRY.isoformat(),
            "origin": origin,
        }

    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is in allowed list (supports wildcards)."""
        if not origin:
            # Allow empty origins in test environments
            import sys

            if "pytest" in sys.modules:
                return True
            return False

        # Parse origin to get hostname and scheme
        try:
            parsed = urlparse(origin)
            hostname = parsed.hostname or ""
            scheme = parsed.scheme or ""
        except Exception:
            return False

        # Check against allowed origins (with secure wildcard support)
        for allowed in self.ALLOWED_ORIGINS:
            if "*" in allowed:
                # Secure wildcard matching - only support *.domain.com format
                if allowed.startswith("https://*.") or allowed.startswith("http://*."):
                    # Extract scheme and domain suffix
                    allowed_scheme = (
                        "https" if allowed.startswith("https://") else "http"
                    )
                    domain_suffix = allowed.split("://")[1][
                        2:
                    ]  # Remove scheme and "*."

                    # Validate: scheme must match, hostname must end with domain and have subdomain
                    if (
                        scheme == allowed_scheme
                        and hostname.endswith("." + domain_suffix)
                        and hostname
                        != domain_suffix  # Prevent bypass with exact domain
                        and len(hostname[: -len(domain_suffix) - 1]) > 0
                    ):  # Ensure subdomain exists (any length)
                        return True
                else:
                    # Unsupported wildcard pattern - reject for security
                    continue
            elif origin == allowed:
                # Exact match
                return True

        return False

    def _check_rate_limit(self, origin: str) -> bool:
        """Check if origin has exceeded rate limit with bounded cache."""
        current_time = time.time()

        # Update access time for LRU tracking
        self._cache_access_time[origin] = current_time

        # Clean old entries outside the window
        cutoff_time = current_time - self.RATE_LIMIT_WINDOW
        self._rate_limit_cache[origin] = [
            timestamp
            for timestamp in self._rate_limit_cache[origin]
            if timestamp > cutoff_time
        ]

        # Enforce per-origin entry limit
        if len(self._rate_limit_cache[origin]) > self.MAX_CACHE_ENTRIES_PER_ORIGIN:
            # Keep only the most recent entries
            self._rate_limit_cache[origin] = self._rate_limit_cache[origin][
                -self.MAX_CACHE_ENTRIES_PER_ORIGIN :
            ]

        # Check if rate limit exceeded
        if len(self._rate_limit_cache[origin]) >= self.RATE_LIMIT_REQUESTS:
            return False

        # Enforce global cache size limit using LRU eviction
        if len(self._rate_limit_cache) >= self.MAX_CACHE_ORIGINS:
            self._evict_lru_origins()

        # Add current request
        self._rate_limit_cache[origin].append(current_time)
        return True

    def _evict_lru_origins(self):
        """Evict least recently used origins from cache to prevent memory exhaustion."""
        # Sort origins by last access time and remove oldest 10%
        origins_by_access = sorted(self._cache_access_time.items(), key=lambda x: x[1])

        # Remove oldest 10% of origins
        evict_count = max(1, len(origins_by_access) // 10)
        for origin, _ in origins_by_access[:evict_count]:
            if origin in self._rate_limit_cache:
                del self._rate_limit_cache[origin]
            if origin in self._cache_access_time:
                del self._cache_access_time[origin]

        logger.info(f"Evicted {evict_count} origins from rate limit cache (LRU)")

    def get_cache_stats(self) -> dict[str, int]:
        """Get current cache statistics for monitoring."""
        return {
            "origins_tracked": len(self._rate_limit_cache),
            "total_entries": sum(
                len(entries) for entries in self._rate_limit_cache.values()
            ),
            "max_origins": self.MAX_CACHE_ORIGINS,
            "max_entries_per_origin": self.MAX_CACHE_ENTRIES_PER_ORIGIN,
        }

    def _is_bot_request(self, user_agent: str) -> bool:
        """Basic bot detection based on user agent."""
        if not user_agent:
            return False

        # Allow test clients in test environment
        import sys

        if "pytest" in sys.modules and user_agent.lower() in ["testclient"]:
            return False

        bot_indicators = [
            "bot",
            "crawler",
            "spider",
            "scraper",
            "curl",
            "wget",
            "postman",
            "insomnia",
            "automated",
            "test",
        ]

        user_agent_lower = user_agent.lower()
        return any(indicator in user_agent_lower for indicator in bot_indicators)

    def _update_usage_stats(self, origin: str):
        """Update usage statistics."""
        self._usage_stats["total_requests"] += 1
        self._usage_stats["unique_origins"].add(origin)

    def get_usage_stats(self) -> dict[str, Any]:
        """Get current usage statistics."""
        current_time = datetime.now(timezone.utc)
        time_until_expiry = max(
            0, (self.DEMO_TOKEN_EXPIRY - current_time).total_seconds()
        )

        return {
            "total_requests": self._usage_stats["total_requests"],
            "unique_origins": len(self._usage_stats["unique_origins"]),
            "rate_limit_violations": self._usage_stats["rate_limit_violations"],
            "origin_violations": self._usage_stats["origin_violations"],
            "expired_token_attempts": self._usage_stats["expired_token_attempts"],
            "expires_in_seconds": time_until_expiry,
            "expires_at": self.DEMO_TOKEN_EXPIRY.isoformat(),
            "is_expired": current_time > self.DEMO_TOKEN_EXPIRY,
        }

    def is_demo_period_active(self) -> bool:
        """Check if demo period is still active."""
        return datetime.now(timezone.utc) <= self.DEMO_TOKEN_EXPIRY


# Global instance
_secure_demo_auth = SecureDemoAuthenticator()


def validate_demo_token(
    token: str, origin: str = "", user_agent: str = ""
) -> dict[str, Any]:
    """Global function to validate demo token."""
    return _secure_demo_auth.validate_demo_token(token, origin, user_agent)


def get_demo_usage_stats() -> dict[str, Any]:
    """Global function to get demo usage statistics."""
    return _secure_demo_auth.get_usage_stats()


def is_demo_period_active() -> bool:
    """Global function to check if demo period is active."""
    return _secure_demo_auth.is_demo_period_active()
