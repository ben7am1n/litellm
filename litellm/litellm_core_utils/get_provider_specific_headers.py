from collections.abc import Sequence
from typing import Final
from urllib.parse import urlparse

from litellm.types.utils import ProviderSpecificHeader


class ProviderSpecificHeaderUtils:
    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None,
        custom_llm_provider: str | None,
        api_base: str | None = None,
    ) -> dict:
        """
        Get the provider specific headers for the given custom llm provider.

        Accepts either a single ProviderSpecificHeader or a sequence of them. Each entry
        carries its own comma-separated provider list, so headers that are safe for several
        providers and headers that are safe for exactly one can travel on the same request
        without sharing a scope. Entries whose provider list does not contain
        `custom_llm_provider` contribute nothing.

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        scoped_headers: Final = (
            (provider_specific_header,) if isinstance(provider_specific_header, dict) else provider_specific_header
        )

        matched_headers: Final = {}
        for scoped_header in scoped_headers:
            stored_providers = scoped_header.get("custom_llm_provider", "")
            provider_list = [p.strip() for p in stored_providers.split(",")]
            if custom_llm_provider in provider_list:
                extra_headers = scoped_header.get("extra_headers", {})
                if custom_llm_provider == "anthropic" and api_base:
                    from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

                    parsed_api_base: Final = urlparse(api_base)
                    is_official_anthropic_api: Final = (
                        parsed_api_base.scheme == "https"
                        and parsed_api_base.hostname == "api.anthropic.com"
                        and parsed_api_base.port in (None, 443)
                    )
                    if not is_official_anthropic_api:
                        extra_headers = {
                            header: value
                            for header, value in extra_headers.items()
                            if not (header.lower() == "authorization" and is_anthropic_oauth_key(value))
                        }
                matched_headers.update(extra_headers)

        return matched_headers
