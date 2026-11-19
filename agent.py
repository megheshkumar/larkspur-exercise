"""Larkspur disruption agent. This is the file you build.

It runs right now, and it is wrong in four places. The trace shows each one
before the code does, so read the trace first:

    python3 run.py K7PQ2M --trace

Where you edit:   grep -n '✏' agent.py   (six marks, one per place)
Steps and gates:  https://anthropicpartnerbasecamp.bts.com/
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from support import (MODEL, SYSTEM_PROMPT, call_local, execute_tool, mcp_client,
                     new_session, record_tool_result, runtime_preamble)

MAX_TOOL_CALLS = 8

TONE_ADDENDUM = ""
EXTRA_TOOLS: List[Dict[str, Any]] = []
LOCAL_TOOLS: Dict[str, Any] = {}


def text_of(response) -> str:
    """Return the last non-empty text block, never content[0]."""
    texts = [b.text for b in response.content
             if getattr(b, "type", None) == "text" and b.text]
    return texts[-1] if texts else ""


def tool_results(response) -> List[Dict[str, Any]]:
    """Run tool calls, normalizing flight dates before dispatch."""
    results = []
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        tool_input = dict(block.input or {})
        if block.name == "get_flight_status":
            value = tool_input.get("date")
            if isinstance(value, str):
                value = value.strip()
                for date_format in ("%Y-%m-%d", "%m/%d/%Y"):
                    try:
                        parsed = datetime.strptime(value, date_format)
                        tool_input["date"] = parsed.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(
                        "get_flight_status date must be YYYY-MM-DD or MM/DD/YYYY; "
                        f"got {value!r}"
                    )

        if block.name in mcp_client.tool_names:
            output = mcp_client.call_remote(block.name, tool_input)
        elif block.name in LOCAL_TOOLS:
            output = call_local(LOCAL_TOOLS[block.name], block.name, tool_input)
        else:
            output = execute_tool(block.name, tool_input)

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(output),
        })
    return results


def run_agent(pnr: str, last_name: str, message: str) -> str:
    """Run the tool loop until Claude stops asking for tools."""
    client, tracer = new_session()
    tools = tool_list()
    messages = [
        {"role": "user", "content": f"PNR {pnr}, last name {last_name}. {message}"},
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=runtime_preamble() + SYSTEM_PROMPT + TONE_ADDENDUM,
        thinking={"type": "adaptive"},
        tools=tools,
        messages=messages,
    )

    turns = 1
    while response.stop_reason == "tool_use" and turns < MAX_TOOL_CALLS:
        assistant_blocks = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif hasattr(block, "model_dump"):
                assistant_blocks.append(block.model_dump(exclude_none=True))

        results = tool_results(response)
        assistant_ids = {
            block["id"] for block in assistant_blocks
            if block.get("type") == "tool_use"
        }
        result_ids = {
            block["tool_use_id"] for block in results
            if block.get("type") == "tool_result"
        }
        if not result_ids or not result_ids.issubset(assistant_ids):
            raise RuntimeError(
                f"Tool ID mismatch: assistant={assistant_ids}, results={result_ids}"
            )

        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": results})
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=runtime_preamble() + SYSTEM_PROMPT + TONE_ADDENDUM,
            thinking={"type": "adaptive"},
            tools=tools,
            messages=messages,
        )
        turns += 1

    return text_of(response)


def tool_list() -> List[Dict[str, Any]]:
    """Offer each tool once, with MCP as the owner of remotely served tools.

    If a locally authored schema or implementation has the same name as an MCP
    tool, keep the MCP tool and remove the local registration. This prevents the
    model from receiving ambiguous duplicate tool names.
    """
    remote_tools = mcp_client.tools()
    remote_names = {tool["name"] for tool in remote_tools if tool.get("name")}

    # MCP is the single owner for any overlapping tool name. Do not leave a
    # stale local implementation registered for dispatch either.
    for name in remote_names:
        LOCAL_TOOLS.pop(name, None)

    local_tools = build_tools() + list(EXTRA_TOOLS)
    offered = []
    seen = set(remote_names)
    for schema in local_tools:
        name = schema.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        offered.append(schema)
    return offered + remote_tools


# The nine given tools. MCP tools are discovered separately and appended by
# tool_list(); they are not duplicated here.
def build_tools() -> List[Dict[str, Any]]:
    """Anthropic-shaped schemas for the nine built-in disruption tools."""
    return [
        {
            "name": "lookup_booking",
            "description": (
                "Retrieve a Larkspur reservation from Altura by confirmation code (PNR) "
                "and the passenger's last name. Both are required to prevent a lookup on "
                "a guessed PNR. Returns fare family, loyalty tier, the segment that needs "
                "attention, and any group/partner/minor/SSR flags relevant to scope."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"pnr": {"type": "string"}, "last_name": {"type": "string"}},
                "required": ["pnr", "last_name"],
            },
        },
        {
            "name": "get_flight_status",
            "description": (
                "Look up a Larkspur or Larkspur Link flight's current OpsFeed status for "
                "one local date: status, delay minutes, and cause. Use this before telling "
                "a customer anything about flight timing; never state it from memory. "
                "The date argument MUST use YYYY-MM-DD. Convert booking dates if needed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "flight_no": {"type": "string"},
                    "date": {
                        "type": "string",
                        "description": "Required format YYYY-MM-DD, e.g. 2025-05-08.",
                        "pattern": r"^\d{4}-\d{2}-\d{2}$",
                    },
                },
                "required": ["flight_no", "date"],
            },
        },
        {
            "name": "search_alternatives",
            "description": (
                "Search for available Larkspur replacement flights for this booking. "
                "Provide the passenger's PNR. Returns candidate flights and option IDs "
                "that can be offered or held for rebooking."
            ),
            "input_schema": {
                "type": "object", "properties": {"pnr": {"type": "string"}},
                "required": ["pnr"],
            },
        },
        {
            "name": "check_policy",
            "description": (
                "Resolve what Larkspur owes this customer for the disruption: rebooking "
                "waiver, refund path, meal/hotel/ground care, goodwill eligibility and cap, "
                "and escalation triggers. Booking attributes are derived from the booking. "
                "Every response carries a policy_row_id; cite it when referring to the decision."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pnr": {"type": "string"},
                    "cause_code": {"type": "string", "enum": ["WX", "ATC", "MX", "CREW", "SEC"]},
                    "delay_minutes": {"type": "integer"},
                    "status": {"type": "string", "enum": ["ON_TIME", "DELAYED", "CANCELLED", "DIVERTED"]},
                    "wait_minutes_for_alternative": {"type": "integer"},
                    "chosen_option_id": {"type": "string"},
                },
                "required": ["pnr", "cause_code", "delay_minutes", "status"],
            },
        },
        {
            "name": "hold_seat",
            "description": "Place a 15-minute hold on one alternative. Reversible; it simply expires.",
            "input_schema": {
                "type": "object",
                "properties": {"option_id": {"type": "string"}, "pnr": {"type": "string"}},
                "required": ["option_id", "pnr"],
            },
        },
        {
            "name": "confirm_rebooking",
            "description": (
                "Finalize a held seat. Irreversible. Requires a confirmation_token that "
                "only the customer's own Confirm-click can produce."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"hold_id": {"type": "string"}, "confirmation_token": {"type": "string"}},
                "required": ["hold_id", "confirmation_token"],
            },
        },
        {
            "name": "issue_voucher",
            "description": (
                "Issue a meal, ground, hotel, or goodwill voucher. Auto-approves within "
                "the policy threshold; above it, returns a pending status for a human. "
                "Always pass the policy_row_id that made it eligible."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "voucher_type": {"type": "string", "enum": ["meal", "ground", "hotel", "goodwill"]},
                    "amount_usd": {"type": "number"},
                    "pnr": {"type": "string"},
                    "policy_row_id": {"type": "string"},
                },
                "required": ["voucher_type", "amount_usd", "pnr", "policy_row_id"],
            },
        },
        {
            "name": "escalate_to_human",
            "description": (
                "Hand this conversation to a human, with your reasoning attached. Use for "
                "groups, partner segments, unaccompanied minors, refunds, or anything else "
                "out of scope."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pnr": {"type": "string"}, "reason": {"type": "string"},
                    "summary_for_human": {"type": "string"}, "queue": {"type": "string"},
                },
                "required": ["pnr", "reason", "summary_for_human"],
            },
        },
        {
            "name": "send_confirmation",
            "description": "Send the customer a written confirmation of what was just done.",
            "input_schema": {
                "type": "object",
                "properties": {"pnr": {"type": "string"}, "message": {"type": "string"}},
                "required": ["pnr", "message"],
            },
        },
    ]
