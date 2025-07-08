import re

_TS_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b|\b\d{4}-\d{2}-\d{2}\b")
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_NUM_RE = re.compile(r"\b\d+\b")


def clean_line(line: str) -> str:
    line = _TS_RE.sub("", line)
    line = _HEX_RE.sub("", line)
    line = _NUM_RE.sub("", line)
    return line.lower().strip()
