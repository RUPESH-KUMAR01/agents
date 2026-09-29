"""
Personal calculator agent (Toy Project 1).

Runtime: Groq (OpenAI-compatible chat + tools).
Learning target: Anthropic Messages API patterns — tool definitions with
input_schema, assistant tool_use blocks, user tool_result blocks, and the
agent loop. Groq field names differ; see comments at each translation point.

Anthropic docs (for when you switch providers):
  https://docs.anthropic.com/en/docs/build-with-claude/tool-use
"""

from __future__ import annotations

import ast
import json
import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_TOOL_ROUNDS = 10

# --- Tool implementations ---------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_safe_eval_expr(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval_expr(node.left), _safe_eval_expr(node.right))
    raise ValueError("Only numbers and + - * / // % ** are allowed")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    tree = ast.parse(expression.strip(), mode="eval")
    result = _safe_eval_expr(tree)
    if result == int(result):
        return str(int(result))
    return str(result)


def get_time(timezone: str = "UTC") -> str:
    """Return current local time in the given IANA timezone (default UTC)."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        raise ValueError(f"Unknown timezone: {timezone}") from None
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


_LENGTH_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
}

_WEIGHT_TO_KG = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "oz": 0.028349523125,
    "ounce": 0.028349523125,
    "ounces": 0.028349523125,
}


def _convert_temperature(value: float, unit_from: str, unit_to: str) -> float:
    f = unit_from.lower()
    t = unit_to.lower()
    if f == t:
        return value
    # Normalize to Celsius
    if f in ("c", "celsius"):
        c = value
    elif f in ("f", "fahrenheit"):
        c = (value - 32) * 5 / 9
    elif f in ("k", "kelvin"):
        c = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {unit_from}")
    if t in ("c", "celsius"):
        return c
    if t in ("f", "fahrenheit"):
        return c * 9 / 5 + 32
    if t in ("k", "kelvin"):
        return c + 273.15
    raise ValueError(f"Unknown temperature unit: {unit_to}")


def convert_units(value: float, unit_from: str, unit_to: str, category: str = "auto") -> str:
    """
    Convert between units. category: length, weight, temperature, or auto (guess).
    """
    cat = category.lower()
    u_from = unit_from.lower()
    u_to = unit_to.lower()

    if cat == "auto":
        if u_from in _LENGTH_TO_METERS and u_to in _LENGTH_TO_METERS:
            cat = "length"
        elif u_from in _WEIGHT_TO_KG and u_to in _WEIGHT_TO_KG:
            cat = "weight"
        elif u_from in ("c", "f", "k", "celsius", "fahrenheit", "kelvin") and u_to in (
            "c",
            "f",
            "k",
            "celsius",
            "fahrenheit",
            "kelvin",
        ):
            cat = "temperature"
        else:
            raise ValueError("Could not infer category; set category to length, weight, or temperature")

    if cat == "length":
        base = value * _LENGTH_TO_METERS[u_from]
        out = base / _LENGTH_TO_METERS[u_to]
    elif cat == "weight":
        base = value * _WEIGHT_TO_KG[u_from]
        out = base / _WEIGHT_TO_KG[u_to]
    elif cat == "temperature":
        out = _convert_temperature(value, u_from, u_to)
    else:
        raise ValueError(f"Unknown category: {category}")

    return str(out)


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "calculate": calculate,
    "get_time": get_time,
    "convert_units": convert_units,
}

# Anthropic shape: name, description, input_schema (JSON Schema).
# Claude receives these in messages.create(tools=[...]).
ANTHROPIC_STYLE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "calculate",
        "description": "Evaluate a arithmetic expression with +, -, *, /, //, %, ** and parentheses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression to evaluate, e.g. '(12 + 4) * 3'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current date and time in an IANA timezone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. UTC, America/New_York, Asia/Kolkata",
                }
            },
            "required": [],
        },
    },
    {
        "name": "convert_units",
        "description": "Convert a numeric value between length, weight, or temperature units.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Numeric amount to convert"},
                "unit_from": {"type": "string", "description": "Source unit, e.g. km, lb, f"},
                "unit_to": {"type": "string", "description": "Target unit, e.g. mi, kg, c"},
                "category": {
                    "type": "string",
                    "enum": ["auto", "length", "weight", "temperature"],
                    "description": "Unit family; auto guesses from unit names",
                },
            },
            "required": ["value", "unit_from", "unit_to"],
        },
    },
]


def anthropic_tools_to_groq(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Groq uses OpenAI's tool wire format. Maps Anthropic tool defs → Groq tools.

    Anthropic:  { name, description, input_schema }
    Groq/OpenAI: { type: "function", function: { name, description, parameters } }
    """
    groq_tools = []
    for t in tools:
        groq_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
        )
    return groq_tools


def execute_tool_use(tool_name: str, tool_input: dict[str, Any]) -> str:
    """
    Run one tool. On Claude, each assistant tool_use block gets a matching
    user message content block: { type: "tool_result", tool_use_id, content }.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return handler(**tool_input)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def run_agent(client: Groq, user_message: str) -> str:
    """
    Agent loop (same control flow as Claude):

      messages = [user question]
      loop:
        response = model(messages, tools)
        if stop_reason == end_turn and no tool_use:
          return text
        for each tool_use:
          result = execute(tool_use)
          messages += assistant(response) + user(tool_results)
    """
    # Groq: list of {role, content}. Claude: same roles; tool results are user content blocks.
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    groq_tools = anthropic_tools_to_groq(ANTHROPIC_STYLE_TOOLS)

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=groq_tools,
            tool_choice="auto",
        )
        choice = response.choices[0]

        assistant_message = choice.message

        # Anthropic: stop_reason "tool_use" vs "end_turn"
        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            return (assistant_message.content or "").strip()

        # Preserve full assistant turn (required before tool results on both APIs)
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            # Anthropic tool_use: { id, name, input }
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}

            content = execute_tool_use(tool_name, tool_input)

            # Anthropic equivalent:
            # { "role": "user", "content": [{ "type": "tool_result",
            #     "tool_use_id": id, "content": content }] }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )

    return "Stopped: too many tool rounds."


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY in .env (project root).", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(run_agent(client, prompt))
        return

    print("Calculator agent (Groq runtime, Anthropic-style tools). Type 'quit' to exit.\n")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in {"quit", "exit", "q"}:
            break
        print(f"Agent: {run_agent(client, line)}\n")


if __name__ == "__main__":
    main()
