from dataclasses import dataclass
import re


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str


INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore (all|previous) instructions", "prompt injection attempt"),
    (r"reveal (the )?system prompt", "system prompt exfiltration"),
    (r"developer message", "system prompt exfiltration"),
    (r"jailbreak", "prompt injection attempt"),
    (r"bypass (security|safety|guardrails)", "guardrail bypass attempt"),
    (r"exfiltrate", "data exfiltration attempt"),
]


def evaluate_prompt_safety(question: str) -> GuardrailResult:
    for pattern, reason in INJECTION_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return GuardrailResult(allowed=False, reason=reason)
    return GuardrailResult(allowed=True, reason="ok")
