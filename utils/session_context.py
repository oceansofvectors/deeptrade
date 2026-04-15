"""Shared trading-session helpers for feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytz


@dataclass(frozen=True)
class SessionConfig:
    timezone: str = "America/Chicago"
    open_hour: int = 17
    open_minute: int = 0
    close_hour: int = 16
    close_minute: int = 0

    @property
    def open_minutes(self) -> int:
        return self.open_hour * 60 + self.open_minute

    @property
    def close_minutes(self) -> int:
        return self.close_hour * 60 + self.close_minute

    @property
    def crosses_midnight(self) -> bool:
        return self.close_minutes <= self.open_minutes

    @property
    def session_duration_minutes(self) -> int:
        duration = self.close_minutes - self.open_minutes
        if duration <= 0:
            duration += 24 * 60
        return duration


def session_config_from_mapping(mapping: dict | None) -> SessionConfig:
    mapping = mapping or {}
    return SessionConfig(
        timezone=str(mapping.get("timezone", "America/Chicago")),
        open_hour=int(mapping.get("open_hour", 17)),
        open_minute=int(mapping.get("open_minute", 0)),
        close_hour=int(mapping.get("close_hour", 16)),
        close_minute=int(mapping.get("close_minute", 0)),
    )


def build_session_context(index: pd.Index, session_config: SessionConfig | dict | None = None) -> dict:
    cfg = session_config if isinstance(session_config, SessionConfig) else session_config_from_mapping(session_config)
    local_index = pd.to_datetime(index, utc=True).tz_convert(pytz.timezone(cfg.timezone))
    minute_of_day = (local_index.hour * 60 + local_index.minute).to_numpy(dtype=int)

    if cfg.crosses_midnight:
        session_anchor = local_index.normalize()
        pre_open_mask = minute_of_day < cfg.open_minutes
        session_anchor = session_anchor.where(~pre_open_mask, session_anchor - pd.Timedelta(days=1))
    else:
        session_anchor = local_index.normalize()

    minutes_since_open = (minute_of_day - cfg.open_minutes) % (24 * 60)
    in_session = minutes_since_open < cfg.session_duration_minutes

    return {
        "config": cfg,
        "local_index": local_index,
        "session_anchor": pd.DatetimeIndex(session_anchor),
        "minute_of_day": minute_of_day,
        "minutes_since_open": minutes_since_open.astype(float),
        "in_session": np.asarray(in_session, dtype=bool),
    }
