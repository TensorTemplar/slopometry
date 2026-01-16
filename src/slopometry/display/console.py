"""Singleton Rich Console instance for consistent output across the application."""

from rich.console import Console

# Single console instance used throughout the application
# This ensures pager context works correctly across modules
console = Console()
