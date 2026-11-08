"""Validate an automation's model proposal and return Scheduler data; execute nothing."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import re

import voluptuous as vol

from homeassistant.core import SupportsResponse, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.helpers import area_registry as ar
from homeassistant.util import dt as dt_util

from .const import DOMAIN

ALLOWED_SERVICES = {
    "light.turn_on", "light.turn_off", "switch.turn_on", "switch.turn_off",
    "script.turn_on", "scene.turn_on", "vacuum.start", "vacuum.return_to_base",
}
PROPOSAL_SCHEMA = vol.Schema({
    vol.Required("name"): vol.All(str, vol.Length(min=1, max=160)),
    vol.Required("entity_ids"): vol.All([str], vol.Length(min=1, max=32)),
    vol.Required("service"): vol.In(ALLOWED_SERVICES),
    vol.Required("delay_minutes"): vol.All(int, vol.Range(min=0, max=525600)),
    vol.Required("time"): str,
    vol.Required("date"): str,
    vol.Required("repeat_type"): vol.In(["single", "daily"]),
})


def prepare_schedule(proposal, *, requested_at: datetime, now: datetime, request: str):
    """Translate a validated proposal to scheduler.add, calculating dates in code."""
    value = PROPOSAL_SCHEMA(proposal)
    if type(value["delay_minutes"]) is not int:
        raise ValueError("Delay must be an integer")
    daily = value["repeat_type"] == "daily"
    if daily and not re.search(r"\b(every day|daily|each day)\b", request, re.I):
        raise ValueError("Daily repetition must be explicitly requested")
    if not daily and re.search(r"\b(every|daily|weekly|repeat)\b", request, re.I):
        raise ValueError("The requested repetition needs clarification")
    local_requested = dt_util.as_local(requested_at)
    if value["delay_minutes"]:
        if value["time"] or value["date"] or daily:
            raise ValueError("A relative delay must be one-off with no clock time or date")
        due = requested_at + timedelta(minutes=value["delay_minutes"])
        due = dt_util.as_local(due)
        # User preference: minute resolution, rounded up so a delay never fires early.
        due = due.replace(second=0, microsecond=0) + timedelta(minutes=1 if due.second or due.microsecond else 0)
    else:
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value["time"]):
            raise ValueError("A valid unambiguous clock time is required")
        hour, minute = map(int, value["time"].split(":"))
        if value["date"]:
            day = datetime.strptime(value["date"], "%Y-%m-%d").date()
        else:
            day = local_requested.date()
        due = datetime.combine(day, datetime.min.time(), local_requested.tzinfo).replace(hour=hour, minute=minute)
        if not value["date"] and due <= local_requested:
            due += timedelta(days=1)
        # Reject non-existent/ambiguous local clock times rather than choosing a DST fold.
        if dt_util.as_local(due.astimezone(dt_util.UTC)).replace(tzinfo=None) != due.replace(tzinfo=None):
            raise ValueError("This local time does not exist due to a clock change")
        if due.replace(fold=0).utcoffset() != due.replace(fold=1).utcoffset():
            raise ValueError("This local time is ambiguous due to a clock change")
    if due <= dt_util.as_local(now):
        raise ValueError("That requested time has already passed; please give a new time")
    actions = [{"service": value["service"], "entity_id": entity_id} for entity_id in dict.fromkeys(value["entity_ids"])]
    data = {
        "name": value["name"], "weekdays": ["daily"],
        "start_date": due.date().isoformat(),
        "repeat_type": "repeat" if daily else "single",
        "timeslots": [{"start": due.strftime("%H:%M"), "actions": actions}],
    }
    return {"status": "ready", "schedule": data, "due_at": due.isoformat(),
            "speech": f"Scheduled {value['name']} for {due.strftime('%Y-%m-%d %H:%M %Z')}{' every day' if daily else ', once'}."}


def named_rooms(text, areas):
    """Match whole area names/aliases, rather than guessing from similar words."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    found = set()
    for area in areas:
        for name in [area.name, *area.aliases]:
            name = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
            if name and f" {name} " in f" {normalized} ":
                found.add(area.id)
    return found


@callback
def async_setup_services(hass):
    if hass.services.has_service(DOMAIN, "prepare_schedule"):
        return

    async def handle(call):
        try:
            calls = call.data["tool_calls"]
            if len(calls) != 1 or calls[0].get("type") != "function":
                raise ValueError("Expected exactly one scheduling tool call")
            function = calls[0]["function"]
            args = json.loads(function["arguments"])
            if function["name"] == "ask_schedule_question":
                question = vol.Schema({vol.Required("question"): vol.All(str, vol.Length(min=1, max=500))})(args)["question"]
                return {"status": "clarify", "speech": question + " Nothing is scheduled yet. Please repeat the schedule with the exact target and time."}
            if function["name"] != "create_schedule":
                raise ValueError("Unexpected tool call")
            value = PROPOSAL_SCHEMA(args)
            domain, service = value["service"].split(".")
            if not hass.services.has_service(domain, service):
                raise ValueError("The proposed action is not available")
            for entity_id in value["entity_ids"]:
                state = hass.states.get(entity_id)
                if not state or entity_id.split(".")[0] != domain:
                    raise ValueError("The proposed target is invalid")
                if not async_should_expose(hass, conversation.DOMAIN, entity_id):
                    raise ValueError("The proposed target is not exposed to this assistant")
                if state.state in ("unknown", "unavailable"):
                    raise ValueError("The proposed target is unavailable")
                if domain == "script":
                    areas = list(ar.async_get(hass).async_list_areas())
                    script_rooms = named_rooms(state.name, areas)
                    requested_rooms = named_rooms(call.data["request"], areas)
                    if script_rooms - requested_rooms:
                        raise ValueError("That script includes rooms you did not request; please choose an exact matching task")
            # Validate with the actual HA service schema, without calling the service.
            schema = hass.services.async_services()[domain][service].schema
            if schema:
                schema({"entity_id": value["entity_ids"]})
            requested_at = dt_util.parse_datetime(call.data["requested_at"])
            if requested_at is None or requested_at.tzinfo is None:
                raise ValueError("A timezone-qualified request timestamp is required")
            return prepare_schedule(value, requested_at=requested_at, now=dt_util.utcnow(), request=call.data["request"])
        except (ValueError, KeyError, TypeError, vol.Invalid) as err:
            return {"status": "error", "speech": f"I haven't scheduled anything. {err}"}

    hass.services.async_register(DOMAIN, "prepare_schedule", handle, schema=vol.Schema({
        vol.Required("tool_calls"): list,
        vol.Required("request"): str,
        vol.Required("requested_at"): str,
    }), supports_response=SupportsResponse.ONLY)
