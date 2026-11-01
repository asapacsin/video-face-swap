"""Role-scoped subprocess environments and defense-in-depth log redaction.

This is *not* a filesystem/network sandbox: use runtime permissions and a
separate reviewer identity/home for actual isolation.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping

# Minimal cross-platform process/runtime environment. Credentials are opt-in.
BASE_VARIABLES = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "USERNAME", "USERPROFILE",
    "HOMEDRIVE", "HOMEPATH", "SYSTEMROOT", "WINDIR", "COMSPEC",
    "PATHEXT", "SYSTEMDRIVE", "APPDATA", "LOCALAPPDATA", "TEMP",
    "TMP", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "PYTHONIOENCODING",
    "VIRTUAL_ENV", "JAVA_HOME", "ANDROID_HOME", "ANDROID_SDK_ROOT",
    "GRADLE_USER_HOME", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
})
ROLE_MODEL_VARIABLES = {
    "executor": frozenset({"AI_NATIVE_EXECUTOR_MODEL", "AI_NATIVE_EXECUTOR_EFFORT"}),
    "reviewer": frozenset({"AI_NATIVE_REVIEWER_MODEL", "AI_NATIVE_REVIEWER_EFFORT"}),
    "gate": frozenset(),
}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET_NAME = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL|PRIVATE)", re.I)
SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)\b((?:api[_-]?key|authorization|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\b(?:sk-(?:proj-)?|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{12,}"),
)


def child_environment(role: str, source: Mapping[str, str],
                      protocol: Mapping[str, str] | None = None) -> dict[str, str]:
    """Pass only runtime, role configuration, and explicitly allowlisted variables.

    Use AI_NATIVE_{ROLE}_ENV_ALLOWLIST=VAR1,VAR2 for credentials a role
    genuinely needs. No wildcard expansion is supported.
    """
    if role not in ROLE_MODEL_VARIABLES:
        raise ValueError("unknown subprocess role: " + role)
    raw = source.get(f"AI_NATIVE_{role.upper()}_ENV_ALLOWLIST", "")
    names = {name.strip() for name in re.split(r"[,;\s]+", raw) if name.strip()}
    if any(not NAME.fullmatch(name) for name in names):
        raise ValueError("invalid subprocess environment allowlist")
    allowed = BASE_VARIABLES | ROLE_MODEL_VARIABLES[role] | names
    result = {key: value for key, value in source.items() if key in allowed}
    profile = source.get("AI_NATIVE_" + role.upper() + "_HOME")
    if profile:
        result["HOME"] = profile
        result["USERPROFILE"] = profile
    # Only the controller can set protocol fields; do not trust parent versions.
    result.update(protocol or {})
    return result


def redact_text(value: str, source: Mapping[str, str] | None = None) -> str:
    if not isinstance(value, str):
        raise TypeError("redact_text expects a string")
    output = value
    # Long known sensitive environment values must never be recorded even if
    # they are not explicitly passed to a subprocess.
    sensitive = {
        v for k, v in (source or os.environ).items()
        if SECRET_NAME.search(k) and isinstance(v, str) and len(v) >= 6
    }
    for secret in sorted(sensitive, key=len, reverse=True):
        output = output.replace(secret, "[REDACTED]")
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            output = pattern.sub(lambda m: m.group(1) + "[REDACTED]", output)
        else:
            output = pattern.sub("[REDACTED]", output)
    return output


def redact_argv(argv: list[str], source: Mapping[str, str] | None = None) -> list[str]:
    """Redact both inline secrets and values passed as the next CLI argument."""
    output: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            output.append("[REDACTED]")
            redact_next = False
            continue
        output.append(redact_text(argument, source))
        flag = argument.split("=", 1)[0].lstrip("-").replace("-", "_")
        sensitive_flag = re.search(
            r"(?:^|_)(?:api_?key|token|secret|password|authorization|auth|credential|private_key)(?:$|_)",
            flag, re.I
        )
        if argument.startswith("-") and "=" not in argument and sensitive_flag:
            redact_next = True
    return output
