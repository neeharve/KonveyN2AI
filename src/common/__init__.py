"""
Common utilities and models for KonveyN2AI multi-agent system.

This module provides shared functionality across all three components:
- Amatya Role Prompter
- Janapada Memory
- Svami Orchestrator

Includes JSON-RPC protocol implementation, shared data models,
configuration management, and inter-service communication utilities.
"""

from .models import (
    AdviceRequest,
    AnswerResponse,
    Document,
    DocumentChunk,
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    QueryRequest,
    SearchQuery,
    SearchRequest,
    SearchResult,
    SearchType,
    Snippet,
)
from .agent_manifest import (
    AgentCapability,
    AgentDiscovery,
    AgentManifest,
    AgentManifestGenerator,
    MethodSchema,
    ParameterSchema,
)

__version__ = "1.0.0"
__all__ = [
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcError",
    "JsonRpcErrorCode",
    "Snippet",
    "SearchRequest",
    "SearchQuery",
    "SearchResult",
    "SearchType",
    "DocumentChunk",
    "Document",
    "AdviceRequest",
    "QueryRequest",
    "AnswerResponse",
    "AgentCapability",
    "AgentDiscovery",
    "AgentManifest",
    "AgentManifestGenerator",
    "MethodSchema",
    "ParameterSchema",
]
