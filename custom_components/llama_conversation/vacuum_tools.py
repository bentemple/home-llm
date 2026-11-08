"""Make the vacuum clean-area tool take every room in a single call.

Home Assistant's ``vacuum__HassVacuumCleanArea`` tool takes one area name, so "clean the
kitchen and the living room" makes the model emit one tool call per room. Valetudo (and most
robot vacuums) replace the room list on every ``vacuum.clean_area`` call, so the second call
cancels the first room. The core intent handler already sends every area it matched in one
service call; it just never sees more than one name.

This module wraps the tool on the resolved API instance: the model sees ``area`` as a list of
names, and a multi-area call is resolved and dispatched as a single ``vacuum.clean_area``
service call. Single-area calls are delegated untouched to the original core tool.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import probatio as _schema
except ImportError:  # Home Assistant without probatio
    import voluptuous as _schema

from homeassistant.components.vacuum import (
    DOMAIN as VACUUM_DOMAIN,
    SERVICE_CLEAN_AREA,
    VacuumEntityFeature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, intent, llm

_LOGGER = logging.getLogger(__name__)

CLEAN_AREA_INTENT = "HassVacuumCleanArea"


def _area_list(value: Any) -> list[str]:
    """Normalize the area argument into a list of names.

    Accepts a list (the declared schema) and a comma-separated string, which is what the
    model produces when it ignores the list type.
    """
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_area_list(item if isinstance(item, str) else str(item)))
        return out
    return []


class MultiAreaVacuumCleanTool(llm.Tool):
    """HassVacuumCleanArea with ``area`` as a list, dispatched as one service call."""

    def __init__(self, original: llm.Tool) -> None:
        """Wrap the core tool, keeping its name, integration and annotations."""
        self.original = original
        self.name = original.name
        self.title = original.title
        self.integration = original.integration
        self.annotations = original.annotations
        self.description = (
            original.description or "Tells a vacuum to clean specific areas"
        ) + (
            " Pass every area the user named in ONE call as a list of area names;"
            " the vacuum cleans them all in a single run."
        )
        self.parameters = _schema.Schema(
            {
                _schema.Required("area"): [str],
                _schema.Optional("name"): str,
            }
        )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> llm.ToolResult:
        """Clean all named areas with one vacuum.clean_area call."""
        args = tool_input.tool_args or {}
        areas = _area_list(args.get("area"))
        name = args.get("name") or None

        if len(areas) <= 1:
            # Nothing to merge: let the untouched core tool handle it (single-area
            # semantics, its own slot validation and preferred area/floor handling).
            delegated: dict[str, Any] = {"area": areas[0] if areas else ""}
            if name:
                delegated["name"] = name
            return await self.original.async_call(
                hass,
                llm.ToolInput(
                    tool_name=self.original.name,
                    tool_args=delegated,
                    id=tool_input.id,
                ),
                llm_context,
            )

        area_reg = ar.async_get(hass)
        preferences = llm.async_get_match_preferences(hass, llm_context)
        matched: list[ar.AreaEntry] = []
        for area_name in areas:
            found = list(intent.find_areas(area_name, area_reg))
            if len(found) > 1 and preferences.area_id:
                found = [a for a in found if a.id == preferences.area_id] or found
            if len(found) > 1 and preferences.floor_id:
                found = [a for a in found if a.floor_id == preferences.floor_id] or found
            if not found:
                raise intent.MatchFailedError(
                    result=intent.MatchTargetsResult(
                        is_match=False,
                        no_match_reason=intent.MatchFailedReason.INVALID_AREA,
                        no_match_name=area_name,
                    ),
                    constraints=intent.MatchTargetsConstraints(area_name=area_name),
                )
            matched.append(found[0])

        constraints = intent.MatchTargetsConstraints(
            name=name,
            domains={VACUUM_DOMAIN},
            features=VacuumEntityFeature.CLEAN_AREA,
            assistant=llm_context.assistant,
        )
        # Prefer the vacuum closest to the first named area, like the core handler does.
        match_preferences = intent.MatchTargetsPreferences(
            area_id=matched[0].id, floor_id=matched[0].floor_id
        )
        match_result = intent.async_match_targets(hass, constraints, match_preferences)
        if not match_result.is_match:
            raise intent.MatchFailedError(
                result=match_result,
                constraints=constraints,
                preferences=match_preferences,
            )

        states = match_result.states
        entity_ids = [state.entity_id for state in states]
        area_ids = [area.id for area in matched]
        try:
            await hass.services.async_call(
                VACUUM_DOMAIN,
                SERVICE_CLEAN_AREA,
                {"entity_id": entity_ids, "cleaning_area_id": area_ids},
                blocking=True,
                context=llm_context.context,
            )
        except Exception:
            _LOGGER.exception(
                "Failed to call %s for areas: %s with vacuums: %s",
                SERVICE_CLEAN_AREA,
                area_ids,
                entity_ids,
            )
            raise intent.IntentHandleError(
                f"Failed to call {SERVICE_CLEAN_AREA} for areas: {area_ids}"
                f" with vacuums: {entity_ids}"
            ) from None

        success: list[dict[str, str]] = [
            {"type": "area", "name": area.name, "id": area.id} for area in matched
        ]
        success.extend(
            {"type": "entity", "name": state.name, "id": state.entity_id}
            for state in states
        )
        return llm.ToolResult(
            data={
                "speech": {},
                "response_type": "action_done",
                "data": {"success": success, "failed": []},
            }
        )


def patch_multi_room_vacuum_cleaning(instance: llm.APIInstance | None) -> None:
    """Replace HassVacuumCleanArea in a resolved API instance, in place.

    The instance object is shared by the prompt renderer, the backend tool formatting and
    the chat log execution, so replacing the tool here covers all three.
    """
    if instance is None:
        return
    for index, tool in enumerate(instance.tools):
        if (
            tool.name.rsplit("__", 1)[-1] == CLEAN_AREA_INTENT
            and not isinstance(tool, MultiAreaVacuumCleanTool)
        ):
            instance.tools[index] = MultiAreaVacuumCleanTool(tool)
