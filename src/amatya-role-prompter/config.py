"""
Configuration management for Amatya Role Prompter service.

Handles environment variables, Google Cloud settings, and service configuration
following KonveyN2AI standards.
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AmataConfig:
    """Configuration for Amatya Role Prompter service."""

    # Google Cloud Configuration
    project_id: str
    location: str

    # Vertex AI Configuration
    model_name: str
    temperature: float
    max_output_tokens: int
    top_k: int
    top_p: float

    # Service Configuration
    service_name: str
    port: int
    log_level: str

    # Environment
    environment: str
    debug: bool

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Google Cloud Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Vertex AI Configuration
        self.model_name = os.getenv("VERTEX_AI_MODEL", "text-bison-001")
        self.temperature = float(os.getenv("VERTEX_AI_TEMPERATURE", "0.2"))
        self.max_output_tokens = int(os.getenv("VERTEX_AI_MAX_TOKENS", "300"))
        self.top_k = int(os.getenv("VERTEX_AI_TOP_K", "40"))
        self.top_p = float(os.getenv("VERTEX_AI_TOP_P", "0.8"))

        # Service Configuration
        self.service_name = os.getenv("SERVICE_NAME", "amatya-role-prompter")
        self.port = int(os.getenv("PORT", "8001"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"

        # Validate required configuration
        self._validate_config()

        logger.info(f"Initialized configuration for {self.service_name}")
        logger.info(f"Project: {self.project_id}, Location: {self.location}")
        logger.info(f"Model: {self.model_name}, Environment: {self.environment}")

    def _validate_config(self):
        """Validate required configuration values."""
        required_fields = [
            ("project_id", self.project_id),
            ("location", self.location),
            ("model_name", self.model_name),
        ]

        missing_fields = []
        for field_name, field_value in required_fields:
            if not field_value:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing_fields)}"
            )

    def get_vertex_ai_config(self) -> dict:
        """Get Vertex AI model configuration."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
