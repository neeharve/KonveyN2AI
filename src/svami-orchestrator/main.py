"""
Svami Orchestrator Service - Entry point for user queries and workflow coordination.

This service serves as the main entry point for the KonveyN2AI multi-agent system,
orchestrating the workflow between Janapada (memory/search) and Amatya (role-based advice)
services to provide comprehensive answers to user queries.

Key Features:
- FastAPI application with Guard-Fort middleware
- JSON-RPC client integration for service communication
- Query orchestration: search → advise → respond
- Agent manifest at /.well-known/agent.json
- Health monitoring and error handling
"""

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Add path for common modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.models import AnswerResponse, JsonRpcError, QueryRequest, Snippet
from common.rpc_client import JsonRpcClient
from guard_fort import init_guard_fort

# Configure logger
logger = logging.getLogger(__name__)

# Global variables for JSON-RPC clients
janapada_client: Optional[JsonRpcClient] = None
amatya_client: Optional[JsonRpcClient] = None

# Connection pool configuration constants
RPC_CLIENT_MAX_CONNECTIONS = 20
RPC_CLIENT_MAX_KEEPALIVE_CONNECTIONS = 10
RPC_CLIENT_KEEPALIVE_EXPIRY = 30.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    global janapada_client, amatya_client

    # Initialize JSON-RPC clients with connection pooling
    janapada_url = os.getenv("JANAPADA_URL", "http://localhost:8001")
    amatya_url = os.getenv("AMATYA_URL", "http://localhost:8002")

    janapada_client = JsonRpcClient(
        janapada_url,
        timeout=30,
        max_retries=3,
        max_connections=RPC_CLIENT_MAX_CONNECTIONS,
        max_keepalive_connections=RPC_CLIENT_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=RPC_CLIENT_KEEPALIVE_EXPIRY,
    )
    amatya_client = JsonRpcClient(
        amatya_url,
        timeout=30,
        max_retries=3,
        max_connections=RPC_CLIENT_MAX_CONNECTIONS,
        max_keepalive_connections=RPC_CLIENT_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=RPC_CLIENT_KEEPALIVE_EXPIRY,
    )

    # Register external services for health monitoring
    await register_external_services()

    print("Svami Orchestrator initialized with:")
    print(f"  Janapada URL: {janapada_url}")
    print(f"  Amatya URL: {amatya_url}")
    print("  Guard-Fort middleware enabled")
    print("  Connection pooling enabled:")
    print(f"    Max connections: {RPC_CLIENT_MAX_CONNECTIONS}")
    print(f"    Max keepalive connections: {RPC_CLIENT_MAX_KEEPALIVE_CONNECTIONS}")
    print(f"    Keepalive expiry: {RPC_CLIENT_KEEPALIVE_EXPIRY}s")
    print("  External services registered for health monitoring")

    yield

    # Shutdown - cleanup connection pools
    print("Svami Orchestrator shutting down...")
    if janapada_client:
        await janapada_client.close()
        print("  Janapada client connections closed")
    if amatya_client:
        await amatya_client.close()
        print("  Amatya client connections closed")


# Initialize FastAPI application
app = FastAPI(
    title="Svami Orchestrator Service",
    description="Entry point for user queries and multi-agent workflow coordination",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Initialize Guard-Fort middleware
guard_fort = init_guard_fort(
    app=app,
    service_name="svami-orchestrator",
    enable_auth=True,
    log_level="INFO",
    log_format="json",
    cors_origins=[
        "https://localhost:3000",
        "https://localhost:8080",
        "https://*.run.app",
        "https://*.vercel.app",
        "https://*.netlify.app",
    ],  # Restrict to trusted domains
    auth_schemes=["Bearer", "ApiKey"],
    allowed_paths=[
        "/health",
        "/health/detailed",
        "/",
        "/docs",
        "/openapi.json",
        "/.well-known/agent.json",
    ],
    security_headers=True,
    enable_metrics=True,
    add_metrics_endpoint=True,
    add_health_endpoint=True,
    add_service_status_endpoint=True,
    debug_mode=False,
)


# Register external services for health monitoring
async def register_external_services():
    """Register Janapada and Amatya services for health monitoring."""
    janapada_url = os.getenv("JANAPADA_URL", "http://localhost:8001")
    amatya_url = os.getenv("AMATYA_URL", "http://localhost:8002")

    guard_fort.register_external_service("janapada", janapada_url)
    guard_fort.register_external_service("amatya", amatya_url)


def get_request_id(request: Request) -> str:
    """Dependency to get request ID from Guard-Fort middleware."""
    return getattr(request.state, "request_id", "unknown")


def generate_request_id() -> str:
    """Generate a unique request ID for testing purposes."""
    return f"req-{uuid.uuid4().hex[:12]}"


def classify_intent(question: str) -> dict:
    """
    Classify user intent to determine if question requires full workflow or quick response.

    Args:
        question: User's input question

    Returns:
        Dict with intent type and confidence level
    """
    question_lower = question.lower().strip()

    # Greeting patterns (18+ variations)
    greeting_patterns = [
        "hi",
        "hello",
        "hey",
        "hiya",
        "howdy",
        "good morning",
        "good afternoon",
        "good evening",
        "what's up",
        "whats up",
        "how are you",
        "how's it going",
        "greetings",
        "salutations",
        "yo",
        "sup",
        "nice to meet you",
        "pleasure to meet you",
    ]

    # Check for exact greeting matches
    if question_lower in greeting_patterns:
        return {"intent": "greeting", "confidence": "high"}

    # Check for greeting starts (short phrases ≤5 words)
    words = question_lower.split()
    if len(words) <= 5 and any(
        question_lower.startswith(pattern) for pattern in greeting_patterns
    ):
        return {"intent": "greeting", "confidence": "high"}

    # Check for conversational patterns
    conversational_patterns = [
        "how are things",
        "how's everything",
        "what's new",
        "nice weather",
        "how's your day",
        "thanks",
        "thank you",
    ]

    if any(pattern in question_lower for pattern in conversational_patterns):
        return {"intent": "conversational", "confidence": "medium"}

    # Default to technical query
    return {"intent": "technical", "confidence": "high"}


def get_conversational_response(question: str, role: str) -> str:
    """
    Generate personalized conversational responses for greetings and small talk.

    Args:
        question: User's input question
        role: User's role context

    Returns:
        Friendly, role-aware response with KonveyN2AI branding and playful personality
    """
    question_lower = question.lower().strip()

    # Role-specific technical expertise for enhanced greetings
    role_expertise = {
        "developer": "development questions, code understanding, and technical guidance",
        "backend_developer": "backend development, APIs, databases, and architecture questions",
        "frontend_developer": "frontend development, UI/UX, and client-side questions",
        "security_engineer": "security, authentication, and best practices questions",
        "devops_specialist": "deployment, infrastructure, and DevOps questions",
        "data_engineer": "data pipelines, analytics, and ML-related questions",
        "qa_engineer": "testing strategies, quality assurance, and automation",
        "technical_writer": "documentation, API docs, and technical writing questions",
    }

    # Get role expertise or default
    expertise = role_expertise.get(role, "technical questions and developer onboarding")

    # Handle specific greeting types with playful personality and KonveyN2AI branding
    if question_lower in ["hi", "hello", "hey", "hiya"]:
        return f"Hi there! 👋 I'm your KonveyN2AI assistant, ready to help with {expertise}. How can I assist you today?"
    elif "morning" in question_lower:
        return f"Good morning! ☀️ I'm your KonveyN2AI assistant, specialized in helping developers get up to speed quickly. Whether you need help with {expertise}, I'm here to help!"
    elif "afternoon" in question_lower:
        return f"Good afternoon! ☀️ I'm your KonveyN2AI assistant, designed to help reduce developer onboarding time through AI-powered knowledge transfer. I can help you with {expertise}. What can I help you with today?"
    elif "evening" in question_lower:
        return f"Good evening! 🌙 I'm your KonveyN2AI assistant, ready to help with {expertise}. How can I assist you this evening?"
    elif "how are you" in question_lower:
        return "I'm doing great, thanks for asking! 😊 I'm here and ready to help you with any developer onboarding questions, code explanations, or technical guidance. What would you like to know?"
    elif any(thanks in question_lower for thanks in ["thanks", "thank you"]):
        return "You're welcome! 😊 I'm always here to help with development questions. Feel free to ask anything about code, onboarding, or technical guidance!"
    elif any(casual in question_lower for casual in ["what's up", "whats up", "sup"]):
        return f"Hey there! 👋 Just here being your friendly KonveyN2AI assistant, ready to help with {expertise}. What's up with your code today?"
    elif "nice to meet you" in question_lower:
        return "Nice to meet you too! 😊 I'm your KonveyN2AI assistant, designed to help reduce developer onboarding time through AI-powered knowledge transfer. I can help you understand code, navigate documentation, and get up to speed on new projects. What can I help you with today?"

    # Generic friendly response with full KonveyN2AI value proposition
    return f"Hello! 😊 I'm your KonveyN2AI assistant, designed to help reduce developer onboarding time through AI-powered knowledge transfer. I can help you understand code, navigate documentation, and get up to speed on projects with {expertise}. What can I help you with today?"


@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Return agent manifest for service discovery and capability description."""
    return {
        "name": "Svami Orchestrator Service",
        "version": "1.0.0",
        "protocol": "json-rpc-2.0",
        "description": "Entry point handling user queries and coordinating workflow between Janapada and Amatya services",
        "methods": {
            "answer": {
                "name": "answer",
                "description": "Answer a user query by orchestrating search and advice generation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's question or query",
                        },
                        "role": {
                            "type": "string",
                            "description": "User role for context (default: developer)",
                            "default": "developer",
                        },
                    },
                    "required": ["question"],
                },
                "return_type": "AnswerResponse",
            }
        },
        "endpoints": {
            "answer": "/answer",
            "health": "/health",
            "metrics": "/metrics",
            "services": "/services",
        },
        "capabilities": [
            {
                "name": "query-orchestration",
                "version": "1.0",
                "description": "Multi-agent query workflow orchestration",
            },
            {
                "name": "json-rpc-2.0",
                "version": "2.0",
                "description": "JSON-RPC 2.0 protocol support",
            },
        ],
        "generated_at": "2025-01-26T10:30:00Z",
    }


async def check_service_health(
    service_name: str, client: JsonRpcClient, timeout: float = 2.0
) -> dict:
    """
    Check the health of a specific service with timeout handling.

    Args:
        service_name: Name of the service to check
        client: JSON-RPC client for the service
        timeout: Timeout in seconds for the health check

    Returns:
        Dictionary with health status information
    """
    try:
        # Try a simple RPC call with short timeout
        response = await asyncio.wait_for(
            client.call(method="health", params={}, id="health-check"), timeout=timeout
        )

        if response.error:
            return {
                "service": service_name,
                "status": "unhealthy",
                "error": response.error.message,
                "response_time_ms": None,
            }

        return {
            "service": service_name,
            "status": "healthy",
            "response_time_ms": f"<{timeout*1000:.0f}",  # Approximate since we don't measure exactly
            "last_check": "just_now",
        }

    except asyncio.TimeoutError:
        return {
            "service": service_name,
            "status": "timeout",
            "error": f"Health check timed out after {timeout}s",
            "response_time_ms": f">{timeout*1000:.0f}",
        }
    except Exception as e:
        return {
            "service": service_name,
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": None,
        }


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Comprehensive health check that monitors all dependent services.

    Returns detailed health information for the orchestrator and its dependencies.
    """
    start_time = asyncio.get_running_loop().time()

    # Basic service status
    health_data = {
        "service": "svami-orchestrator",
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "dependencies": {},
    }

    # Check if RPC clients are initialized
    if not janapada_client or not amatya_client:
        health_data["status"] = "degraded"
        health_data["error"] = "RPC clients not initialized"
        health_data["dependencies"] = {
            "janapada": {"status": "not_initialized"},
            "amatya": {"status": "not_initialized"},
        }
        return JSONResponse(status_code=503, content=health_data)

    # Check downstream services in parallel
    janapada_check, amatya_check = await asyncio.gather(
        check_service_health("janapada", janapada_client, timeout=2.0),
        check_service_health("amatya", amatya_client, timeout=2.0),
        return_exceptions=True,
    )

    # Handle exceptions from health checks
    if isinstance(janapada_check, Exception):
        janapada_check = {
            "service": "janapada",
            "status": "error",
            "error": str(janapada_check),
        }

    if isinstance(amatya_check, Exception):
        amatya_check = {
            "service": "amatya",
            "status": "error",
            "error": str(amatya_check),
        }

    health_data["dependencies"]["janapada"] = janapada_check
    health_data["dependencies"]["amatya"] = amatya_check

    # Determine overall health status
    unhealthy_services = []
    for service_name, service_health in health_data["dependencies"].items():
        if service_health["status"] not in ["healthy"]:
            unhealthy_services.append(service_name)

    if unhealthy_services:
        if len(unhealthy_services) == len(health_data["dependencies"]):
            health_data["status"] = "unhealthy"
            status_code = 503
        else:
            health_data["status"] = "degraded"
            status_code = 200  # Partial functionality still available

        health_data["unhealthy_services"] = unhealthy_services
    else:
        status_code = 200

    # Add performance metrics
    total_time = (asyncio.get_running_loop().time() - start_time) * 1000
    health_data["health_check_duration_ms"] = round(total_time, 2)

    return JSONResponse(status_code=status_code, content=health_data)


@app.get("/auth/demo-status")
async def demo_token_status():
    """
    Get current demo token status and usage statistics.

    Returns information about demo token expiry, usage stats, and validation status.
    """
    try:
        # Import here to avoid circular imports
        from guard_fort.secure_demo_auth import (
            get_demo_usage_stats,
            is_demo_period_active,
        )

        stats = get_demo_usage_stats()
        is_active = is_demo_period_active()

        return JSONResponse(
            content={
                "demo_period_active": is_active,
                "token_expires_at": stats["expires_at"],
                "expires_in_seconds": stats["expires_in_seconds"],
                "usage_statistics": {
                    "total_requests": stats["total_requests"],
                    "unique_origins": stats["unique_origins"],
                    "rate_limit_violations": stats["rate_limit_violations"],
                    "origin_violations": stats["origin_violations"],
                    "expired_token_attempts": stats["expired_token_attempts"],
                },
                "status": "active" if is_active else "expired",
                "message": (
                    "Demo token is active and accepting requests from authorized origins"
                    if is_active
                    else "Demo period has ended"
                ),
            }
        )
    except Exception as e:
        logger.error(f"Error getting demo token status: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get demo token status", "details": str(e)},
        )


def handle_rpc_error(error: JsonRpcError, request_id: str) -> AnswerResponse:
    """Handle JSON-RPC errors and generate a friendly error response.

    Args:
        error: The JSON-RPC error object
        request_id: Request ID for tracing

    Returns:
        AnswerResponse with error message
    """
    error_messages = {
        -32000: "I'm experiencing technical difficulties connecting to our services.",
        -32001: "Authentication failed when accessing internal services.",
        -32002: "Access denied to required internal services.",
        -32003: "Invalid request format received.",
        -32004: "External service is currently unavailable.",
        -32005: "Request timed out while processing your query.",
    }

    # Get friendly error message or use generic one
    friendly_message = error_messages.get(
        error.code, "I encountered an unexpected error while processing your request."
    )

    # Add context if available
    if error.data and isinstance(error.data, dict):
        context = error.data.get("reason", "")
        if context:
            friendly_message += f" Details: {context}"

    return AnswerResponse(
        answer=f"I'm sorry, but {friendly_message} Please try again in a moment.",
        sources=[],
        request_id=request_id,
    )


@app.post("/answer", response_model=AnswerResponse)
async def answer_query(
    query: QueryRequest, request_id: str = Depends(get_request_id)
) -> AnswerResponse:
    """
    Answer a user query by orchestrating the multi-agent workflow.

    This endpoint coordinates between Janapada (search) and Amatya (advice) services
    to provide comprehensive answers to user queries.

    Args:
        query: The user's query containing question and role
        request_id: Request ID from Guard-Fort middleware for tracing

    Returns:
        AnswerResponse containing the generated answer, sources, and request ID

    Raises:
        HTTPException: If there are validation errors or service failures
    """
    try:
        # Validate that we have the required clients
        if not janapada_client or not amatya_client:
            raise HTTPException(
                status_code=503,
                detail="Service is not ready - internal services not initialized",
            )

        # FAST-PATH ROUTING: Handle greetings and conversational queries immediately
        intent_classification = classify_intent(query.question)

        if intent_classification["intent"] in ["greeting", "conversational"]:
            print(
                f"[{request_id}] Fast-path: Detected {intent_classification['intent']} with {intent_classification['confidence']} confidence"
            )

            # Generate immediate conversational response (sub-millisecond)
            conversational_answer = get_conversational_response(
                query.question, query.role
            )

            return AnswerResponse(
                answer=conversational_answer,
                sources=[],  # No sources needed for greetings
                request_id=request_id,
            )

        # ORCHESTRATION WORKFLOW: query → search → advise → respond
        # This implements the complete multi-agent workflow as specified in Task 8.3

        # Step 1: Call Janapada to search for relevant snippets
        print(f"[{request_id}] Step 1: Searching for relevant snippets via Janapada...")

        snippets = []
        sources = []

        try:
            search_response = await janapada_client.call(
                method="search", params={"query": query.question, "k": 5}, id=request_id
            )

            # Handle search errors with improved error propagation
            if search_response.error:
                error_msg = f"Search service error: {search_response.error.message}"
                print(
                    f"[{request_id}] Janapada search failed: {search_response.error.message}"
                )
                print(f"[{request_id}] Error code: {search_response.error.code}")
                if search_response.error.data:
                    print(f"[{request_id}] Error details: {search_response.error.data}")
                print(f"[{request_id}] Continuing with graceful degradation...")

                # Store error for potential user feedback
                search_error = error_msg
            else:
                search_error = None
                # Process search results
                if search_response.result and "snippets" in search_response.result:
                    snippet_data = search_response.result["snippets"]
                    print(
                        f"[{request_id}] Found {len(snippet_data)} snippets from Janapada"
                    )

                    # Convert to Snippet objects and collect sources
                    for snippet_dict in snippet_data:
                        try:
                            snippet = Snippet(**snippet_dict)
                            snippets.append(snippet)
                            if snippet.file_path not in sources:
                                sources.append(snippet.file_path)
                        except Exception as e:
                            print(
                                f"[{request_id}] Warning: Invalid snippet format: {e}"
                            )
                            continue
                else:
                    print(f"[{request_id}] No snippets returned from Janapada")

        except Exception as e:
            search_error = f"Search service unavailable: {str(e)}"
            print(f"[{request_id}] Janapada service unavailable: {str(e)}")
            print(f"[{request_id}] Exception type: {type(e).__name__}")
            print(
                f"[{request_id}] Continuing with graceful degradation (no code snippets)..."
            )

        # If no snippets found, provide a graceful response with error details
        if not snippets:
            base_message = "I couldn't find any relevant code snippets for your query."

            # Add error context if available
            if "search_error" in locals() and search_error:
                detailed_message = f"{base_message} There was an issue with the search service: {search_error}. Please try again in a moment."
            else:
                detailed_message = f"{base_message} This might be because the knowledge base is still being populated or your query needs to be more specific. Please try rephrasing your question or check back later."

            return AnswerResponse(
                answer=detailed_message,
                sources=[],
                request_id=request_id,
            )

        # Step 2: Call Amatya to generate advice based on snippets
        print(f"[{request_id}] Step 2: Generating role-based advice via Amatya...")
        advise_response = await amatya_client.call(
            method="advise",
            params={
                "role": query.role,
                "question": query.question,
                "chunks": [snippet.model_dump() for snippet in snippets],
            },
            id=request_id,
        )

        # Handle advice generation errors with improved error propagation
        if advise_response.error:
            error_msg = f"Advice service error: {advise_response.error.message}"
            print(
                f"[{request_id}] Amatya advice failed: {advise_response.error.message}"
            )
            print(f"[{request_id}] Error code: {advise_response.error.code}")
            if advise_response.error.data:
                print(f"[{request_id}] Error details: {advise_response.error.data}")

            # Graceful degradation: return search results with error context
            fallback_answer = (
                f"I found {len(snippets)} relevant code snippets for your question about '{query.question}'. "
                f"However, there was an issue generating role-specific advice: {error_msg}. "
                "Please review the source files for relevant implementation details, or try again in a moment."
            )

            return AnswerResponse(
                answer=fallback_answer, sources=sources, request_id=request_id
            )

        # Step 3: Format and return the final answer
        if advise_response.result and "advice" in advise_response.result:
            answer = advise_response.result["advice"]
            print(f"[{request_id}] Successfully generated complete response")

            return AnswerResponse(answer=answer, sources=sources, request_id=request_id)
        else:
            # Fallback if advice response format is unexpected
            fallback_answer = (
                f"I found {len(snippets)} relevant code snippets for your question. "
                "The advice generation completed but returned an unexpected format. "
                "Please review the source files for implementation details."
            )

            return AnswerResponse(
                answer=fallback_answer, sources=sources, request_id=request_id
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Convert unexpected errors to HTTP exceptions
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing query: {str(e)}",
        ) from e


async def main():
    """Main function to run the Svami orchestrator service."""
    # Configuration
    # Use 0.0.0.0 for development to allow external connections
    # In production, this should be configured via environment variables
    host = os.getenv("SVAMI_HOST", "0.0.0.0")  # nosec B104
    port = int(os.getenv("SVAMI_PORT", "8003"))

    # Run the application
    config = uvicorn.Config(
        app=app, host=host, port=port, log_level="info", reload=False
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
