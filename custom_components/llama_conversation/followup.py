"""Follow-up handlers (local patch "followup1", not upstream).

Something other than the model sometimes asks the user a question: a sentence-trigger automation that found an
ambiguous encyclopedia topic says "Mercury: the planet, the element or the Roman god. Which one did you mean?" The
user's reply ("the planet") then arrives here, at the conversation agent, because it matches no sentence trigger.
Without help the model gets it with no idea what the question was.

``llama_conversation.request_followup`` lets whoever asked claim the next reply in that conversation. It names a
handler, which is any action that returns a response (normally a script), and optional data for it (what was asked).
When the next message in that conversation reaches this agent, the handler gets the reply before the model does and
decides what happens, by returning ``status``:

- ``done``: its ``speech`` is the answer. The follow-up is over and the next message goes to the model as usual.
- ``continue``: its ``speech`` is said and the handler also gets the reply after that one (with its new ``data``,
  if it returned any). The response asks Assist to keep listening, so a satellite or the app opens the mic again.
- ``pass``: the reply was not for the handler ("turn off the kitchen lights"). The follow-up is over and the model
  answers the reply as if nothing had been pending, with the device, satellite and area context it always has.

A handler that fails, times out or returns anything else counts as ``pass``. A follow-up nobody replies to expires
after its timeout. Whatever happens, at most one handler call is made per user message, and a handler only ever
sees messages from the conversation it asked in.

The conversation is the one the action is called from: an automation started by a sentence trigger runs inside
the conversation turn that triggered it, so it needs no id (the app's first message does not carry one yet).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationInput, ConversationResult
from homeassistant.components.conversation.chat_log import current_chat_log
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv, intent

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

SERVICE_REQUEST_FOLLOWUP = "request_followup"
ATTR_HANDLER = "handler"
ATTR_DATA = "data"
ATTR_CONVERSATION_ID = "conversation_id"
ATTR_TIMEOUT = "timeout"

STATUS_DONE = "done"
STATUS_CONTINUE = "continue"
STATUS_PASS = "pass"

DEFAULT_TIMEOUT = 120.0
# The user is waiting for an answer while the handler runs.
HANDLER_TIMEOUT = 20.0

DATA_FOLLOWUPS = f"{DOMAIN}_followups"

REQUEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_HANDLER): cv.service,
        vol.Optional(ATTR_DATA, default=dict): dict,
        vol.Optional(ATTR_CONVERSATION_ID): cv.string,
        vol.Optional(ATTR_TIMEOUT, default=DEFAULT_TIMEOUT): vol.All(vol.Coerce(float), vol.Range(min=5, max=600)),
    }
)


@dataclass
class Followup:
    handler: str
    data: dict[str, Any]
    timeout: float
    expires: float


def _pending(hass: HomeAssistant) -> dict[str, Followup]:
    return hass.data.setdefault(DATA_FOLLOWUPS, {})


@callback
def async_request_followup(
    hass: HomeAssistant, conversation_id: str, handler: str, data: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """Hand the next reply in this conversation to ``handler`` (replaces any follow-up already pending there)."""
    pending = _pending(hass)
    now = time.monotonic()
    for cid in [c for c, f in pending.items() if f.expires <= now]:
        del pending[cid]
    pending[conversation_id] = Followup(handler, dict(data or {}), timeout, now + timeout)


@callback
def async_take_followup(hass: HomeAssistant, conversation_id: str) -> Followup | None:
    """Remove and return the follow-up pending in this conversation, if it has not expired."""
    followup = _pending(hass).pop(conversation_id, None)
    if followup is None or followup.expires <= time.monotonic():
        return None
    return followup


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register llama_conversation.request_followup (once per Home Assistant run)."""
    if hass.services.has_service(DOMAIN, SERVICE_REQUEST_FOLLOWUP):
        return

    async def _async_handle(call: ServiceCall) -> None:
        conversation_id = call.data.get(ATTR_CONVERSATION_ID)
        if not conversation_id and (chat_log := current_chat_log.get()) is not None:
            conversation_id = chat_log.conversation_id
        if not conversation_id:
            raise ServiceValidationError(
                "No conversation to follow up: call this from an automation started by a conversation"
                " (a sentence trigger), or pass conversation_id"
            )
        handler = call.data[ATTR_HANDLER]
        if not hass.services.has_service(*handler.split(".", 1)):
            # A plain validation error (not ServiceNotFound): the caller's trace shows it, and an automation step
            # with continue_on_error still continues.
            raise ServiceValidationError(f"Follow-up handler {handler} does not exist (create it, or fix the name)")
        async_request_followup(hass, conversation_id, handler, call.data[ATTR_DATA], call.data[ATTR_TIMEOUT])
        _LOGGER.debug("Follow-up in %s goes to %s first", conversation_id, call.data[ATTR_HANDLER])

    hass.services.async_register(DOMAIN, SERVICE_REQUEST_FOLLOWUP, _async_handle, schema=REQUEST_SCHEMA)


async def async_handle_followup(
    hass: HomeAssistant, user_input: ConversationInput, chat_log: conversation.ChatLog,
) -> ConversationResult | None:
    """Give the user's message to the handler pending in this conversation.

    Returns the result to answer with, or None when the model should answer (nothing pending, or the handler passed
    or failed). The user's message is already in the chat log either way.
    """
    followup = async_take_followup(hass, chat_log.conversation_id)
    if followup is None:
        return None

    domain, service = followup.handler.split(".", 1)
    payload = {
        "reply": user_input.text,
        "data": followup.data,
        "conversation_id": chat_log.conversation_id,
        "device_id": user_input.device_id,
        "satellite_id": user_input.satellite_id,
        "language": user_input.language,
    }
    try:
        async with asyncio.timeout(HANDLER_TIMEOUT):
            response = await hass.services.async_call(
                domain, service, payload, blocking=True, return_response=True, context=user_input.context
            )
    except Exception as err:  # noqa: BLE001 - a broken handler must never break the conversation
        _LOGGER.warning("Follow-up handler %s failed, the model answers instead: %r", followup.handler, err)
        return None

    status = response.get("status") if isinstance(response, dict) else None
    speech = response.get("speech") if isinstance(response, dict) else None
    if status == STATUS_PASS:
        _LOGGER.debug("Follow-up handler %s passed %r to the model", followup.handler, user_input.text)
        return None
    if status not in (STATUS_DONE, STATUS_CONTINUE) or not isinstance(speech, str):
        _LOGGER.warning(
            "Follow-up handler %s returned %r; expected {status: done|continue|pass, speech: text}. The model answers"
            " instead", followup.handler, response,
        )
        return None

    if status == STATUS_CONTINUE:
        data = response.get("data")
        async_request_followup(
            hass, chat_log.conversation_id, followup.handler,
            data if isinstance(data, dict) else followup.data, followup.timeout,
        )
    chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(agent_id=user_input.agent_id, content=speech)
    )
    response_obj = intent.IntentResponse(language=user_input.language)
    response_obj.async_set_speech(speech)
    return ConversationResult(
        response=response_obj,
        conversation_id=chat_log.conversation_id,
        continue_conversation=status == STATUS_CONTINUE,
    )
