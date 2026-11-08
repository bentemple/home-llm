"""Prompt pre-warm (local patch "prewarm1", not upstream).

When a voice satellite wakes up, the request this integration will send a few seconds later is already known up to
the user's words: the system prompt (device states, time, area sentence) and the tool list. The
``llama_conversation.prewarm`` action sends exactly that prefix to the backend with generation disabled, so the
server processes the long, slow part of a new conversation while the user is still speaking. When the real request
arrives, only the user turn is processed.

What the pre-warm is, and is not:
- It is the real first-turn request minus the user message, rendered by the same code paths (prompt template,
  Home Assistant's system prompt composition, tool formatting, message formatting), sent with ``max_tokens: 0`` and
  ``add_generation_prompt: false``. No text is added to any prompt. Nothing is generated.
- It never touches a chat log or a chat session: Home Assistant's conversation history, traces and chat log
  subscribers never see it, and this integration's per-conversation prompt cache is not written.
- It never blocks or fails the caller: the action only schedules a background task and returns. Every error is
  logged at debug level and dropped.

Only the OpenAI-compatible chat backends (``llama_cpp_server``, ``generic_openai``) support it, because only those
send the request shape this was validated with (llama.cpp server; see eval/results/prewarm/PREWARM.md).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant, ServiceCall, callback
from homeassistant.helpers import config_validation as cv, entity_registry as er, llm
from homeassistant.util import dt as dt_util

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

SERVICE_PREWARM = "prewarm"
ATTR_AGENT_ID = "agent_id"
ATTR_SATELLITE_ID = "satellite_id"
ATTR_DEVICE_ID = "device_id"
ATTR_LANGUAGE = "language"
ATTR_TIMEOUT = "timeout"
ATTR_KEEP_WARM = "keep_warm"

# The request must outlive the server's work: llama.cpp cancels a request whose client disconnects, so a timeout
# shorter than the prompt processing (about 20 s for a cold 9k-token prompt) would throw the work away.
DEFAULT_TIMEOUT = 60.0

PREWARM_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_AGENT_ID): cv.entity_id,
        vol.Optional(ATTR_SATELLITE_ID): cv.entity_id,
        vol.Optional(ATTR_DEVICE_ID): cv.string,
        vol.Optional(ATTR_LANGUAGE): cv.string,
        vol.Optional(ATTR_TIMEOUT, default=DEFAULT_TIMEOUT): vol.All(vol.Coerce(float), vol.Range(min=1, max=300)),
        vol.Optional(ATTR_KEEP_WARM, default=False): cv.boolean,
    }
)

# Home Assistant 2026.10 keys (homeassistant/helpers/chat_session.py, homeassistant/helpers/entity_component.py).
_DATA_CHAT_SESSION = "chat_session"
_DATA_ENTITY_COMPONENTS = "entity_components"


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register llama_conversation.prewarm (once per Home Assistant run)."""
    if hass.services.has_service(DOMAIN, SERVICE_PREWARM):
        return

    async def _async_handle(call: ServiceCall) -> None:
        # Never raise into the caller (an automation on the satellite): schedule and return.
        try:
            _async_schedule_prewarm(hass, dict(call.data))
        except Exception:  # noqa: BLE001
            _LOGGER.debug("prewarm: not scheduled", exc_info=True)

    hass.services.async_register(DOMAIN, SERVICE_PREWARM, _async_handle, schema=PREWARM_SCHEMA)


@callback
def _async_schedule_prewarm(hass: HomeAssistant, data: dict[str, Any]) -> None:
    from homeassistant.components.conversation.agent_manager import async_get_agent
    from .conversation import LocalLLMAgent

    agent_id = data[ATTR_AGENT_ID]
    agent = async_get_agent(hass, agent_id)
    if not isinstance(agent, LocalLLMAgent):
        _LOGGER.debug("prewarm: %s is not a Local LLMs conversation agent, skipped", agent_id)
        return

    satellite_id = data.get(ATTR_SATELLITE_ID)
    device_id = data.get(ATTR_DEVICE_ID)
    if device_id is None and satellite_id:
        entry = er.async_get(hass).async_get(satellite_id)
        device_id = entry.device_id if entry else None

    key = device_id or satellite_id or ""
    if key in agent.prewarm_in_flight:
        _LOGGER.debug("prewarm: one is already running for %s, skipped", key)
        return

    timeout = float(data.get(ATTR_TIMEOUT, DEFAULT_TIMEOUT))
    keep_warm = bool(data.get(ATTR_KEEP_WARM, False))
    language = data.get(ATTR_LANGUAGE) or hass.config.language

    async def _run() -> None:
        agent.prewarm_in_flight.add(key)
        try:
            async with asyncio.timeout(timeout + 5):
                result = await agent.async_prewarm(device_id, satellite_id, language, timeout, keep_warm)
            _LOGGER.debug("prewarm: %s (satellite %s, device %s)", result, satellite_id, device_id)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("prewarm: failed (satellite %s, device %s)", satellite_id, device_id, exc_info=True)
        finally:
            agent.prewarm_in_flight.discard(key)

    hass.async_create_background_task(_run(), name=f"{DOMAIN} prewarm {key}")


def satellite_conversation_is_active(hass: HomeAssistant, satellite_id: str | None) -> str | None:
    """The conversation id a wake-up of this satellite will continue, or None when it will start a new one.

    An assist satellite keeps the id of its last conversation and hands it to the next pipeline run; Home Assistant
    continues that conversation while its chat session is younger than CONVERSATION_TIMEOUT (5 minutes since the
    last turn), otherwise it starts a new one (assist_satellite/entity.py, helpers/chat_session.py).
    """
    if not satellite_id:
        return None
    from homeassistant.helpers.chat_session import CONVERSATION_TIMEOUT

    component = hass.data.get(_DATA_ENTITY_COMPONENTS, {}).get("assist_satellite")
    entity = component.get_entity(satellite_id) if component else None
    conversation_id = getattr(entity, "_conversation_id", None)
    if not conversation_id:
        return None
    session = hass.data.get(_DATA_CHAT_SESSION, {}).get(conversation_id)
    if session is None or session.last_updated + CONVERSATION_TIMEOUT < dt_util.utcnow():
        return None
    return conversation_id


def active_satellite_conversations(hass: HomeAssistant) -> list[str]:
    """Conversation ids of every assist satellite that is still inside its conversation window."""
    component = hass.data.get(_DATA_ENTITY_COMPONENTS, {}).get("assist_satellite")
    if not component:
        return []
    ids = [satellite_conversation_is_active(hass, entity.entity_id) for entity in component.entities]
    return [i for i in ids if i]


def llm_context_for(hass: HomeAssistant, device_id: str | None, language: str) -> llm.LLMContext:
    """The LLM context a pipeline run from this device gives the agent (ConversationInput.as_llm_context)."""
    return llm.LLMContext(
        platform=DOMAIN,
        context=Context(),
        language=language,
        assistant=conversation.DOMAIN,
        device_id=device_id,
    )


async def compose_system_prompt(
    hass: HomeAssistant,
    llm_context: llm.LLMContext,
    llm_api: llm.APIInstance | None,
    user_llm_prompt: str | None,
    extra_system_prompt: str | None,
) -> str:
    """The system message ChatLog.async_provide_llm_data builds (Home Assistant 2026.10.4, chat_log.py:752-787),
    without the side effects: no chat log, chat session, subscriber notification or trace.

    It runs on a detached ChatLog that is never registered, so the template expansion is Home Assistant's own code.
    `llm_api` must be the instance the chat log would hold (PreparedLLMAPI's, whose api_prompt is blank).
    """
    log = conversation.ChatLog(hass, "llama_conversation_prewarm")   # detached: not in any chat session
    user_name: str | None = None
    if llm_context.context and llm_context.context.user_id and (
        user := await hass.auth.async_get_user(llm_context.context.user_id)
    ):
        user_name = user.name

    parts = [
        await log._async_expand_prompt_template(   # noqa: SLF001
            llm_context, (user_llm_prompt or llm.DEFAULT_INSTRUCTIONS_PROMPT), llm_context.language, user_name
        )
    ]
    if llm_api:
        parts.append(llm_api.api_prompt)
    tools: list[llm.Tool] = llm_api.tools if llm_api else []
    if not any(tool.name.endswith("GetDateTime") for tool in tools):
        parts.append(
            await log._async_expand_prompt_template(   # noqa: SLF001
                llm_context, llm.DATE_TIME_PROMPT, llm_context.language, user_name
            )
        )
    if extra_system_prompt:
        parts.append(extra_system_prompt)
    return "\n".join(parts)


class PrewarmRecord:
    """What the last pre-warm for a device sent, to compare with the real request (debug log only)."""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.monotonic = time.monotonic()


def describe_match(prewarm: PrewarmRecord | None, real_system_prompt: str) -> str:
    """One debug line: did the real first turn start with the pre-warmed system prompt?"""
    if prewarm is None:
        return "no pre-warm for this device"
    age = time.monotonic() - prewarm.monotonic
    a, b = prewarm.system_prompt, real_system_prompt
    if a == b:
        return f"system prompt identical to the pre-warm sent {age:.1f} s earlier ({len(a)} chars): the server reuses it"
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return (f"system prompt differs from the pre-warm sent {age:.1f} s earlier at char {i} of {len(b)} "
            f"(pre-warm {a[max(0, i - 40):i + 40]!r} / real {b[max(0, i - 40):i + 40]!r}): the server reuses the part before it")
