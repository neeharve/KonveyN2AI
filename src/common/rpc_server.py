"""
JSON-RPC 2.0 Server implementation for KonveyN2AI multi-agent system.

This module provides server-side utilities for handling JSON-RPC requests,
including method registration, request validation, error handling, and response formatting.
"""

import asyncio
import json
import logging
import traceback
from inspect import Parameter, signature
from typing import Any, Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .models import JsonRpcError, JsonRpcErrorCode, JsonRpcRequest, JsonRpcResponse
from .agent_manifest import AgentManifestGenerator

logger = logging.getLogger(__name__)


class JsonRpcMethodRegistry:
    """Registry for JSON-RPC method handlers."""

    def __init__(self) -> None:
        self._methods: dict[str, Callable[..., Any]] = {}
        self._method_schemas: dict[str, dict[str, Any]] = {}

    def register(
        self, method_name: str, handler: Callable[..., Any], description: str = ""
    ) -> None:
        """Register a method handler with optional description."""
        if method_name in self._methods:
            raise ValueError(f"Method '{method_name}' is already registered")

        if method_name.startswith("rpc."):
            raise ValueError("Method names starting with 'rpc.' are reserved")

        self._methods[method_name] = handler
        self._method_schemas[method_name] = self._extract_method_schema(
            handler, description
        )
        logger.info(f"Registered JSON-RPC method: {method_name}")

    def get_handler(self, method_name: str) -> Optional[Callable[..., Any]]:
        """Get a registered method handler."""
        return self._methods.get(method_name)

    def get_methods(self) -> dict[str, dict[str, Any]]:
        """Get all registered methods with their schemas."""
        return self._method_schemas.copy()

    def _extract_method_schema(
        self, handler: Callable[..., Any], description: str
    ) -> dict[str, Any]:
        """Extract method schema from handler function signature."""
        sig = signature(handler)
        params = {}

        for param_name, param in sig.parameters.items():
            # Skip 'self' and context parameters
            if param_name in ("self", "request_id", "context"):
                continue

            param_info = {
                "name": param_name,
                "required": param.default == Parameter.empty,
                "type": str(param.annotation)
                if param.annotation != Parameter.empty
                else "Any",
            }
            params[param_name] = param_info

        return {
            "description": description
            or handler.__doc__
            or f"Handler for {handler.__name__}",
            "parameters": params,
            "handler": handler.__name__,
        }


class JsonRpcServer:
    """JSON-RPC 2.0 Server implementation."""

    def __init__(
        self,
        title: str = "JSON-RPC Server",
        version: str = "1.0.0",
        description: str = "",
    ):
        self.title = title
        self.version = version
        self.description = description
        self.registry = JsonRpcMethodRegistry()
        self._middleware: list[Callable[..., Any]] = []
        self._manifest_generator = AgentManifestGenerator(title, version, description)

    def method(
        self, name: str, description: str = ""
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a JSON-RPC method."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.registry.register(name, func, description)
            self._manifest_generator.register_method(name, func, description)
            return func

        return decorator

    def add_middleware(self, middleware: Callable[..., Any]) -> None:
        """Add middleware function that processes requests/responses."""
        self._middleware.append(middleware)

    async def handle_request(self, request: Request) -> Response:
        """Handle incoming JSON-RPC request."""
        request_id = None

        try:
            # Parse request body
            body = await request.body()
            request_data = json.loads(body.decode("utf-8"))

            # Handle batch requests
            if isinstance(request_data, list):
                if len(request_data) == 0:
                    return self._create_error_response(
                        None,
                        JsonRpcErrorCode.INVALID_REQUEST,
                        "Invalid Request: Empty batch",
                    )

                responses = []
                for req_data in request_data:
                    response = await self._process_single_request(req_data, request)
                    if response:  # Don't include None responses (notifications)
                        responses.append(response.model_dump())

                return JSONResponse(content=responses if responses else None)

            # Handle single request
            response = await self._process_single_request(request_data, request)
            if response is None:
                # Notification request - no response
                return Response(status_code=204)

            return JSONResponse(content=response.model_dump())

        except json.JSONDecodeError:
            return self._create_error_response(
                request_id, JsonRpcErrorCode.PARSE_ERROR, "Parse error: Invalid JSON"
            )
        except Exception as e:
            logger.exception("Unexpected error handling JSON-RPC request")
            return self._create_error_response(
                request_id, JsonRpcErrorCode.INTERNAL_ERROR, f"Internal error: {str(e)}"
            )

    async def _process_single_request(
        self, request_data: dict[str, Any], http_request: Request
    ) -> Optional[JsonRpcResponse]:
        """Process a single JSON-RPC request."""
        request_id = request_data.get("id")

        try:
            # Validate request structure
            rpc_request = JsonRpcRequest(**request_data)
            request_id = rpc_request.id

            # Apply middleware
            for middleware in self._middleware:
                try:
                    await middleware(rpc_request, http_request)
                except Exception as e:
                    logger.warning(f"Middleware error: {e}")

            # Get method handler
            handler = self.registry.get_handler(rpc_request.method)
            if not handler:
                if rpc_request.id is None:
                    return None  # Don't respond to notification for unknown method
                return JsonRpcResponse.create_error(
                    rpc_request.id,
                    JsonRpcError(
                        code=JsonRpcErrorCode.METHOD_NOT_FOUND,
                        message=f"Method not found: {rpc_request.method}",
                        data=None,
                    ),
                )

            # Prepare handler arguments
            handler_kwargs = rpc_request.params.copy()

            # Add context parameters if handler expects them
            handler_sig = signature(handler)
            if "request_id" in handler_sig.parameters:
                handler_kwargs["request_id"] = rpc_request.id
            if "context" in handler_sig.parameters:
                handler_kwargs["context"] = {
                    "http_request": http_request,
                    "rpc_request": rpc_request,
                }

            # Call handler
            try:
                if asyncio.iscoroutinefunction(handler):
                    # Async function or method
                    result = await handler(**handler_kwargs)
                else:
                    # Sync function or method
                    result = handler(**handler_kwargs)

                # Handle notification requests (no id field)
                if rpc_request.id is None:
                    return None

                return JsonRpcResponse.create_success(rpc_request.id, result)

            except TypeError as e:
                # Parameter validation error
                if rpc_request.id is None:
                    return None  # Don't respond to notification with invalid params
                return JsonRpcResponse.create_error(
                    rpc_request.id,
                    JsonRpcError(
                        code=JsonRpcErrorCode.INVALID_PARAMS,
                        message=f"Invalid params: {str(e)}",
                        data={"error": str(e)},
                    ),
                )
            except Exception as e:
                # Handler execution error
                logger.exception(f"Error executing handler {rpc_request.method}")
                if rpc_request.id is None:
                    return None  # Don't respond to notification with handler errors
                return JsonRpcResponse.create_error(
                    rpc_request.id,
                    JsonRpcError(
                        code=JsonRpcErrorCode.INTERNAL_ERROR,
                        message=f"Handler error: {str(e)}",
                        data={
                            "traceback": traceback.format_exc()
                            if logger.isEnabledFor(logging.DEBUG)
                            else None
                        },
                    ),
                )

        except ValidationError as e:
            # Pydantic validation error
            return JsonRpcResponse.create_error(
                request_id or "unknown",
                JsonRpcError(
                    code=JsonRpcErrorCode.INVALID_REQUEST,
                    message="Invalid Request: Validation failed",
                    data={"validation_errors": e.errors()},
                ),
            )
        except Exception as e:
            # Unexpected error
            logger.exception("Unexpected error processing JSON-RPC request")
            return JsonRpcResponse.create_error(
                request_id or "unknown",
                JsonRpcError(
                    code=JsonRpcErrorCode.INTERNAL_ERROR,
                    message=f"Internal error: {str(e)}",
                    data=None,
                ),
            )

    def _create_error_response(
        self, request_id: Optional[str], code: int, message: str
    ) -> JSONResponse:
        """Create an error response as JSONResponse."""
        error_response = JsonRpcResponse.create_error(
            request_id or "unknown", JsonRpcError(code=code, message=message, data=None)
        )
        return JSONResponse(content=error_response.model_dump())

    def get_manifest(self) -> dict[str, Any]:
        """Get the enhanced agent manifest for /.well-known/agent.json endpoint."""
        return self._manifest_generator.to_dict()

    def add_capability(self, name: str, version: str, description: str) -> None:
        """Add a custom capability to the agent manifest."""
        from .agent_manifest import AgentCapability

        capability = AgentCapability(
            name=name, version=version, description=description
        )
        self._manifest_generator.add_capability(capability)

    def add_endpoint(self, name: str, url: str) -> None:
        """Add a service endpoint to the agent manifest."""
        self._manifest_generator.add_endpoint(name, url)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata in the agent manifest."""
        self._manifest_generator.set_metadata(key, value)


# Utility functions for common response patterns


def create_success_response(request_id: str, result: Any) -> JsonRpcResponse:
    """Create a successful JSON-RPC response."""
    return JsonRpcResponse.create_success(request_id, result)


def create_error_response(
    request_id: str, code: int, message: str, data: Optional[dict[str, Any]] = None
) -> JsonRpcResponse:
    """Create an error JSON-RPC response."""
    error = JsonRpcError(code=code, message=message, data=data)
    return JsonRpcResponse.create_error(request_id, error)


# Middleware utilities


async def logging_middleware(
    rpc_request: JsonRpcRequest, http_request: Request
) -> None:
    """Middleware to log JSON-RPC requests."""
    logger.info(f"JSON-RPC call: {rpc_request.method} (id: {rpc_request.id})")


async def request_id_propagation_middleware(
    rpc_request: JsonRpcRequest, http_request: Request
) -> None:
    """Middleware to propagate request IDs in HTTP headers."""
    # Add request ID to response headers (handled by FastAPI middleware)
    pass


# Decorator for easier method registration without server instance

_global_server = None


def rpc_method(
    name: str, description: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Global decorator to register JSON-RPC methods."""
    global _global_server
    if _global_server is None:
        _global_server = JsonRpcServer()

    return _global_server.method(name, description)


def get_global_server() -> JsonRpcServer:
    """Get the global JSON-RPC server instance."""
    global _global_server
    if _global_server is None:
        _global_server = JsonRpcServer()
    return _global_server
