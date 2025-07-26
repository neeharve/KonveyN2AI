"""
Amatya Role Prompter - FastAPI application for LLM-powered advice generation.

This service provides role-specific prompting and advice generation using
Google Cloud Vertex AI text models. It implements JSON-RPC 2.0 protocol
for communication with other KonveyN2AI components.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import common modules
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from common.rpc_server import JsonRpcServer
from common.models import AdviceRequest, Snippet
from guard_fort import GuardFort

# Import service modules
try:
    from .advisor import AdvisorService
    from .config import AmataConfig
except ImportError:
    from advisor import AdvisorService
    from config import AmataConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for service components
advisor_service: AdvisorService = None
config: AmataConfig = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global advisor_service, config

    logger.info("Starting Amatya Role Prompter service...")

    try:
        # Initialize configuration
        config = AmataConfig()
        logger.info("Configuration loaded successfully")

        # Initialize advisor service
        advisor_service = AdvisorService(config)
        await advisor_service.initialize()
        logger.info("Advisor service initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    finally:
        # Cleanup
        if advisor_service:
            await advisor_service.cleanup()
        logger.info("Amatya Role Prompter service stopped")


# Create FastAPI application
app = FastAPI(
    title="Amatya Role Prompter",
    description="LLM-powered advice generation with role-specific prompting",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GuardFort security middleware
guard_fort = GuardFort(
    app,
    service_name="amatya-role-prompter",
    enable_auth=True,
    allowed_paths=["/health", "/", "/docs", "/openapi.json", "/.well-known/agent.json"],
)

# Create JSON-RPC server
rpc_server = JsonRpcServer(
    title="Amatya Role Prompter",
    version="1.0.0",
    description="LLM-powered advice generation with role-specific prompting",
)


@rpc_server.method("advise", "Generate role-specific advice based on code snippets")
async def advise(role: str, chunks: list[dict], request_id: str = None) -> dict:
    """
    Generate role-specific advice based on provided code snippets.

    Args:
        role: User role (e.g., 'backend_developer', 'security_engineer')
        chunks: List of code snippets with file_path and content
        request_id: Optional request ID for tracking

    Returns:
        dict: Response containing generated advice
    """
    try:
        # Convert chunks to Snippet objects
        snippet_objects = [Snippet(**chunk) for chunk in chunks]

        # Create advice request
        advice_request = AdviceRequest(role=role, chunks=snippet_objects)

        # Check if advisor service is available
        if advisor_service is None:
            # Create a temporary advisor service for testing
            from config import AmataConfig
            from advisor import AdvisorService

            temp_config = AmataConfig()
            temp_advisor = AdvisorService(temp_config)
            await temp_advisor.initialize()

            advice = await temp_advisor.generate_advice(advice_request)
            await temp_advisor.cleanup()
        else:
            # Use the global advisor service
            advice = await advisor_service.generate_advice(advice_request)

        logger.info(f"Generated advice for role '{role}' with {len(chunks)} chunks")

        return {"answer": advice}

    except Exception as e:
        logger.error(f"Error generating advice: {e}")
        raise


# JSON-RPC endpoint
@app.post("/")
async def json_rpc_endpoint(request: Request):
    """Main JSON-RPC 2.0 endpoint."""
    return await rpc_server.handle_request(request)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring."""
    try:
        # Check if advisor service is ready
        if advisor_service:
            is_healthy = await advisor_service.is_healthy()
            if is_healthy:
                return {
                    "status": "healthy",
                    "service": "amatya-role-prompter",
                    "version": "1.0.0",
                    "mode": "vertex_ai" if advisor_service.llm_model else "mock",
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "service": "amatya-role-prompter",
                        "version": "1.0.0",
                        "error": "Advisor service not ready",
                    },
                )
        else:
            # Service not initialized yet (e.g., during startup or testing)
            return {
                "status": "starting",
                "service": "amatya-role-prompter",
                "version": "1.0.0",
                "message": "Service is starting up",
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "amatya-role-prompter",
                "version": "1.0.0",
                "error": str(e),
            },
        )


# Agent manifest endpoint
@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Agent discovery manifest endpoint."""
    return {
        "name": "Amatya Role Prompter",
        "description": "LLM-powered advice generation with role-specific prompting",
        "version": "1.0.0",
        "methods": [
            {
                "name": "advise",
                "description": "Generate role-specific advice based on code snippets",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "User role (e.g., 'backend_developer', 'security_engineer')",
                        },
                        "chunks": {
                            "type": "array",
                            "description": "Code snippets for context",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["file_path", "content"],
                            },
                        },
                    },
                    "required": ["role", "chunks"],
                },
            }
        ],
        "endpoints": {
            "rpc": "/",
            "health": "/health",
            "manifest": "/.well-known/agent.json",
        },
    }


# Development server
if __name__ == "__main__":
    # Use 0.0.0.0 for development to allow external connections
    # In production, this should be configured via environment variables
    host = os.getenv("HOST", "0.0.0.0")  # nosec B104
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")
