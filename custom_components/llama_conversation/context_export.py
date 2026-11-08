"""Render a fresh automation-owned prompt, optionally warming that exact prefix.

This action never runs tools or writes to the agent's conversation history/cache.
"""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, entity_registry as er, llm
from homeassistant.util import dt as dt_util

from .const import (
    CONF_CHAT_MODEL, CONF_ENABLE_LEGACY_TOOL_CALLING,
    CONF_PROMPT, DEFAULT_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_PROMPT, DOMAIN,
)
from .prewarm import compose_system_prompt
from .utils import PreparedLLMAPI, get_oai_formatted_tools, prompt_as_literal

SCHEMA = vol.Schema({
    vol.Required("agent_id"): cv.entity_id,
    vol.Optional("device_id"): cv.string,
    vol.Optional("satellite_id"): cv.entity_id,
    vol.Optional("language"): cv.string,
    vol.Optional("extra_system_prompt", default=""): cv.string,
    vol.Optional("include_tools", default=True): cv.boolean,
    vol.Optional("additional_tools", default=[]): [{
        vol.Required("type"): vol.In(["function"]),
        vol.Required("function"): {
            vol.Required("name"): cv.string,
            vol.Optional("description"): cv.string,
            vol.Required("parameters"): dict,
            vol.Optional("strict"): cv.boolean,
        },
    }],
    vol.Optional("prewarm", default=False): cv.boolean,
    vol.Optional("timeout", default=60): vol.All(vol.Coerce(float), vol.Range(min=1, max=300)),
})


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register a response-only action, distinct from the wake-word prewarm."""
    if hass.services.has_service(DOMAIN, "get_context"):
        return

    async def handle(call: ServiceCall) -> dict[str, Any]:
        from homeassistant.components.conversation.agent_manager import async_get_agent
        from .backends.generic_openai import GenericOpenAIAPIClient
        from .conversation import LocalLLMAgent

        data = call.data
        agent = async_get_agent(hass, data["agent_id"])
        if not isinstance(agent, LocalLLMAgent):
            raise HomeAssistantError("agent_id must select a Local LLMs conversation agent")
        device_id = data.get("device_id")
        satellite_id = data.get("satellite_id")
        if satellite_id:
            entity = er.async_get(hass).async_get(satellite_id)
            satellite_device = entity.device_id if entity else None
            if not satellite_device:
                raise HomeAssistantError("The satellite has no registered device")
            if device_id and device_id != satellite_device:
                raise HomeAssistantError("device_id and satellite_id refer to different devices")
            device_id = satellite_device

        options = agent.runtime_options
        context = llm.LLMContext(
            platform=DOMAIN, context=call.context,
            language=data.get("language") or hass.config.language,
            assistant=conversation.DOMAIN, device_id=device_id,
        )
        api = None
        if api_id := options.get(CONF_LLM_HASS_API):
            api = await llm.async_get_api(hass, api_id, context)

        # Render from current HA states, even if a live conversation has a cached prompt.
        rendered = agent.client._generate_system_prompt(
            options.get(CONF_PROMPT, DEFAULT_PROMPT), api, options,
        )
        prepared = await PreparedLLMAPI(api).async_get_api_instance(context) if api else None
        prompt = await compose_system_prompt(
            hass, context, prepared, prompt_as_literal(rendered), data["extra_system_prompt"] or None,
        )
        use_tools = data["include_tools"] and not options.get(
            CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
        )
        tools = get_oai_formatted_tools(prepared, agent.client._async_get_all_exposed_domains()) if prepared and use_tools else []
        tools = tools + list(data["additional_tools"])
        names = [tool["function"]["name"] for tool in tools]
        if len(set(names)) != len(names):
            raise HomeAssistantError("Tool names must be unique")
        result: dict[str, Any] = {
            "system_prompt": prompt,
            "messages": [{"role": "system", "content": prompt}],
            "tools": tools,
            "model": options[CONF_CHAT_MODEL],
            "device_id": device_id,
            "rendered_at": dt_util.utcnow().isoformat(),
            "prewarmed": False,
        }
        if data["prewarm"]:
            if not isinstance(agent.client, GenericOpenAIAPIClient):
                raise HomeAssistantError("Prewarm requires the OpenAI-compatible chat backend")
            # Use exactly the tool selection returned to the caller. Descriptions in
            # the rendered prompt remain context; no tool can execute on this path.
            result["timings"] = await agent.client.async_prewarm(
                prompt, prepared if use_tools else None, options, data["timeout"],
                tools_override=tools,
            )
            result["prewarmed"] = True
        return result

    hass.services.async_register(
        DOMAIN, "get_context", handle, schema=SCHEMA, supports_response=SupportsResponse.ONLY,
    )
