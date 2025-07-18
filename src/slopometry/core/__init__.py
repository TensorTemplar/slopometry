"""Core infrastructure shared between solo and summoner features."""

from slopometry.core.database import EventDatabase
from slopometry.core.settings import settings

__all__ = [
    "EventDatabase",
    "settings",
]
