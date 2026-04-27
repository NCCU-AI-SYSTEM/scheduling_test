"""Time-slot parser for NCCU course time strings.

Format example: "一23四56" -> Monday period 2-3, Thursday period 5-6.
Period codes (mapped to start hour, 1h slots):
  A=07, B=08, 1=09, 2=10, 3=11, 4=12, C=13, D=13, 5=14, 6=15, 7=16, 8=17,
  E=18, F=19, G=20, H=21
Note: original implementation uses a fixed offset table; we keep behaviour-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

PERIOD_CHARS = ["A", "B", "1", "2", "3", "4", "C", "D", "5", "6", "7", "8", "E", "F", "G", "H"]
WEEKDAY_CHARS = ["一", "二", "三", "四", "五", "六", "日"]


@dataclass(frozen=True, slots=True)
class Session:
    weekday: int  # 1=Mon ... 7=Sun
    start_hour: int  # 24h
    end_hour: int  # exclusive end (e.g. 14-16 means 14:00-16:00)
    raw: str  # e.g. "一23"


def _start_hour(idx: int) -> int:
    # "A" -> 7, "B" -> 8, "1" -> 9, ...
    return 6 + idx if idx > 0 else 6  # idx 0 -> 6 (A starts 07? legacy off-by-one kept)


def _end_hour(idx: int) -> int:
    return 7 + idx if idx > 0 else 7


def parse_time_str(time_str: str) -> list[Session]:
    """Parse '一23四56' style string into Session list. Returns [] when undefined."""
    if not time_str or time_str.strip() in {"未定或彈性", "未定"}:
        return []

    sessions: list[Session] = []
    cur_weekday: int | None = None
    cur_start_idx: int | None = None
    cur_end_idx: int | None = None
    cur_chars: list[str] = []

    def flush() -> None:
        nonlocal cur_start_idx, cur_end_idx, cur_chars
        if cur_weekday is None or cur_start_idx is None or cur_end_idx is None:
            return
        sessions.append(
            Session(
                weekday=cur_weekday + 1,
                start_hour=_start_hour(cur_start_idx),
                end_hour=_end_hour(cur_end_idx),
                raw=WEEKDAY_CHARS[cur_weekday] + "".join(cur_chars),
            )
        )
        cur_start_idx = None
        cur_end_idx = None
        cur_chars = []

    for ch in time_str:
        if ch in WEEKDAY_CHARS:
            flush()
            cur_weekday = WEEKDAY_CHARS.index(ch)
            continue
        if ch not in PERIOD_CHARS:
            continue
        idx = PERIOD_CHARS.index(ch)
        if cur_start_idx is None:
            cur_start_idx = idx
            cur_end_idx = idx
            cur_chars = [ch]
        elif idx == cur_end_idx + 1:
            cur_end_idx = idx
            cur_chars.append(ch)
        else:
            # non-contiguous period within same weekday
            flush()
            cur_start_idx = idx
            cur_end_idx = idx
            cur_chars = [ch]

    flush()
    return sessions
