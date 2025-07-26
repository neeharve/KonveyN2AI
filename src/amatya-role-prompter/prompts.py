"""
Role-specific prompt construction for advice generation.

This module handles the construction of prompts tailored to different user roles,
incorporating code snippets and context for optimal LLM responses.
"""

import logging
from typing import List, Dict
from common.models import Snippet

logger = logging.getLogger(__name__)


class PromptConstructor:
    """Constructs role-specific prompts for advice generation."""

    def __init__(self):
        """Initialize the prompt constructor."""
        self.role_templates = self._load_role_templates()
        logger.info("PromptConstructor initialized")

    def construct_prompt(self, role: str, chunks: List[Snippet]) -> str:
        """
        Construct a role-specific prompt for advice generation.

        Args:
            role: User role (e.g., 'backend_developer', 'security_engineer')
            chunks: List of code snippets for context

        Returns:
            str: Constructed prompt for the LLM
        """
        # Build context from chunks
        context = self._build_context(chunks)

        # Get role-specific template
        template = self._get_role_template(role)

        # Construct the final prompt
        prompt = template.format(
            role=role.replace("_", " ").title(), context=context, num_files=len(chunks)
        )

        logger.info(f"Constructed prompt for role '{role}' with {len(chunks)} chunks")
        return prompt

    def _build_context(self, chunks: List[Snippet]) -> str:
        """
        Build context string from code snippets.

        Args:
            chunks: List of code snippets

        Returns:
            str: Formatted context string
        """
        if not chunks:
            return "No code snippets provided."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"**File {i}: {chunk.file_path}**\n```\n{chunk.content}\n```"
            )

        return "\n\n".join(context_parts)

    def _get_role_template(self, role: str) -> str:
        """
        Get the appropriate template for a role.

        Args:
            role: User role

        Returns:
            str: Template string for the role
        """
        # Normalize role name
        normalized_role = role.lower().replace(" ", "_")

        # Return specific template if available, otherwise use default
        return self.role_templates.get(normalized_role, self.role_templates["default"])

    def _load_role_templates(self) -> Dict[str, str]:
        """Load role-specific prompt templates."""
        return {
            "default": """You are an experienced onboarding assistant helping a new {role} understand and contribute to this project.

You have access to the following code excerpts and documentation:

{context}

Based on this information, provide a comprehensive onboarding guide for the {role}. Your response should:

1. **Overview**: Briefly explain what this project does and the {role}'s role in it
2. **Getting Started**: Step-by-step guidance for setting up and understanding the codebase
3. **Key Areas**: Highlight the most important files and concepts for this role
4. **Best Practices**: Share relevant coding standards and patterns used in this project
5. **Next Steps**: Suggest specific tasks or areas to explore further

Format your response in clear markdown with:
- Numbered lists for step-by-step instructions
- Code references using backticks
- File path references for each recommendation
- At least 3 specific file references from the provided context

Be concise but thorough, focusing on actionable guidance that will help the {role} become productive quickly.""",
            "backend_developer": """You are a senior backend developer helping onboard a new backend developer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced backend developer, provide a comprehensive onboarding guide focusing on:

1. **Architecture Overview**: Explain the backend architecture, API design, and data flow
2. **Development Setup**: Guide through environment setup, dependencies, and local development
3. **Core Backend Components**: Highlight key services, models, and business logic
4. **API Patterns**: Explain REST/GraphQL patterns, authentication, and error handling
5. **Database & Storage**: Cover data models, migrations, and storage patterns
6. **Testing Strategy**: Explain unit tests, integration tests, and testing patterns
7. **Deployment & DevOps**: Cover CI/CD, containerization, and deployment processes

Include specific file references and code examples. Focus on backend-specific concerns like performance, scalability, and maintainability.""",
            "frontend_developer": """You are a senior frontend developer helping onboard a new frontend developer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced frontend developer, provide a comprehensive onboarding guide focusing on:

1. **UI/UX Architecture**: Explain the frontend architecture, component structure, and design system
2. **Development Environment**: Guide through setup, build tools, and development workflow
3. **Component Library**: Highlight reusable components, styling patterns, and UI conventions
4. **State Management**: Explain data flow, state management patterns, and API integration
5. **Routing & Navigation**: Cover page structure, routing patterns, and user flows
6. **Performance**: Discuss optimization strategies, bundling, and loading patterns
7. **Testing**: Explain component testing, e2e testing, and testing utilities

Include specific file references and focus on frontend-specific concerns like user experience, accessibility, and browser compatibility.""",
            "security_engineer": """You are a senior security engineer helping onboard a new security engineer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced security engineer, provide a comprehensive onboarding guide focusing on:

1. **Security Architecture**: Explain authentication, authorization, and security boundaries
2. **Threat Model**: Identify potential security risks and mitigation strategies
3. **Authentication & Authorization**: Cover user management, tokens, and access controls
4. **Data Protection**: Explain encryption, data handling, and privacy considerations
5. **Input Validation**: Highlight validation patterns and injection prevention
6. **Security Testing**: Cover security testing tools, penetration testing, and vulnerability scanning
7. **Compliance**: Discuss relevant standards, regulations, and audit requirements

Include specific file references and focus on security-specific concerns like threat vectors, secure coding practices, and incident response.""",
            "devops_engineer": """You are a senior DevOps engineer helping onboard a new DevOps engineer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced DevOps engineer, provide a comprehensive onboarding guide focusing on:

1. **Infrastructure Overview**: Explain the deployment architecture, cloud services, and infrastructure as code
2. **CI/CD Pipeline**: Detail the build, test, and deployment processes
3. **Containerization**: Cover Docker, Kubernetes, and container orchestration
4. **Monitoring & Logging**: Explain observability, metrics, alerts, and log aggregation
5. **Security & Compliance**: Cover infrastructure security, secrets management, and compliance
6. **Scaling & Performance**: Discuss auto-scaling, load balancing, and performance optimization
7. **Disaster Recovery**: Explain backup strategies, failover, and incident response

Include specific file references and focus on operational concerns like reliability, scalability, and maintainability.""",
            "data_scientist": """You are a senior data scientist helping onboard a new data scientist to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced data scientist, provide a comprehensive onboarding guide focusing on:

1. **Data Architecture**: Explain data sources, pipelines, and storage systems
2. **ML/AI Components**: Highlight machine learning models, training pipelines, and inference systems
3. **Data Processing**: Cover ETL processes, data cleaning, and feature engineering
4. **Model Development**: Explain model training, validation, and experimentation workflows
5. **Data Analysis**: Discuss analytics tools, visualization, and reporting systems
6. **Model Deployment**: Cover model serving, monitoring, and A/B testing
7. **Data Governance**: Explain data quality, privacy, and compliance requirements

Include specific file references and focus on data-specific concerns like model performance, data quality, and reproducibility.""",
            "product_manager": """You are a senior product manager helping onboard a new product manager to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced product manager, provide a comprehensive onboarding guide focusing on:

1. **Product Overview**: Explain the product vision, user personas, and key features
2. **Technical Architecture**: Understand the system design and technical constraints
3. **User Journey**: Map out user flows and interaction patterns
4. **Feature Prioritization**: Explain how features are planned and prioritized
5. **Metrics & Analytics**: Cover key performance indicators and measurement strategies
6. **Stakeholder Management**: Identify key stakeholders and communication patterns
7. **Release Process**: Understand deployment cycles and feature rollout strategies

Include specific file references and focus on product-specific concerns like user experience, business impact, and technical feasibility.""",
            "qa_engineer": """You are a senior QA engineer helping onboard a new QA engineer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced QA engineer, provide a comprehensive onboarding guide focusing on:

1. **Testing Strategy**: Explain the overall testing approach and methodologies
2. **Test Automation**: Cover automated testing frameworks and CI/CD integration
3. **Test Coverage**: Identify critical paths and edge cases to test
4. **Quality Metrics**: Understand quality gates and acceptance criteria
5. **Bug Tracking**: Explain defect management and reporting processes
6. **Performance Testing**: Cover load testing and performance benchmarks
7. **Security Testing**: Identify security testing requirements and tools

Include specific file references and focus on quality-specific concerns like test coverage, reliability, and user experience validation.""",
            "technical_writer": """You are a senior technical writer helping onboard a new technical writer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced technical writer, provide a comprehensive onboarding guide focusing on:

1. **Documentation Architecture**: Explain the documentation structure and organization
2. **Content Strategy**: Understand the target audiences and content types
3. **API Documentation**: Cover API reference and integration guides
4. **User Guides**: Identify user-facing documentation needs
5. **Developer Documentation**: Explain technical documentation for developers
6. **Content Management**: Understand the documentation workflow and tools
7. **Style Guidelines**: Cover writing standards and consistency requirements

Include specific file references and focus on documentation-specific concerns like clarity, accuracy, and maintainability.""",
        }

    def get_available_roles(self) -> List[str]:
        """Get list of available role templates."""
        return list(self.role_templates.keys())

    def get_context_aware_prompt(self, role: str, chunks: List[Snippet]) -> str:
        """
        Get a context-aware prompt that adapts based on the code content.

        Args:
            role: User role
            chunks: List of code snippets

        Returns:
            str: Context-aware prompt
        """
        # Analyze the code content to determine project characteristics
        project_context = self._analyze_project_context(chunks)

        # Get base template
        base_prompt = self.construct_prompt(role, chunks)

        # Add context-specific guidance
        if project_context["has_api"]:
            base_prompt += "\n\n**API Integration Notes**: This project includes API endpoints. Pay special attention to request/response handling, authentication, and error management."

        if project_context["has_database"]:
            base_prompt += "\n\n**Database Integration Notes**: This project includes database operations. Consider data modeling, migrations, and query optimization."

        if project_context["has_auth"]:
            base_prompt += "\n\n**Security Notes**: This project includes authentication/authorization. Focus on security best practices and access control patterns."

        if project_context["has_tests"]:
            base_prompt += "\n\n**Testing Notes**: This project includes test files. Review the testing strategy and ensure new code follows established patterns."

        return base_prompt

    def _analyze_project_context(self, chunks: List[Snippet]) -> Dict[str, bool]:
        """
        Analyze code chunks to understand project characteristics.

        Args:
            chunks: List of code snippets

        Returns:
            dict: Project characteristics
        """
        context = {
            "has_api": False,
            "has_database": False,
            "has_auth": False,
            "has_tests": False,
            "has_frontend": False,
            "has_docker": False,
        }

        for chunk in chunks:
            content_lower = chunk.content.lower()
            file_path_lower = chunk.file_path.lower()

            # Check for API patterns
            if any(
                keyword in content_lower
                for keyword in ["@app.", "fastapi", "flask", "router", "endpoint"]
            ):
                context["has_api"] = True

            # Check for database patterns
            if any(
                keyword in content_lower
                for keyword in [
                    "database",
                    "db.",
                    "session",
                    "query",
                    "model",
                    "sqlalchemy",
                ]
            ):
                context["has_database"] = True

            # Check for authentication patterns
            if any(
                keyword in content_lower
                for keyword in ["auth", "login", "token", "jwt", "password", "security"]
            ):
                context["has_auth"] = True

            # Check for test files
            if any(
                keyword in file_path_lower
                for keyword in ["test_", "_test", "tests/", "spec_"]
            ):
                context["has_tests"] = True

            # Check for frontend patterns
            if any(
                keyword in content_lower
                for keyword in ["react", "vue", "angular", "component", "jsx", "tsx"]
            ):
                context["has_frontend"] = True

            # Check for Docker
            if "dockerfile" in file_path_lower or "docker" in content_lower:
                context["has_docker"] = True

        return context
