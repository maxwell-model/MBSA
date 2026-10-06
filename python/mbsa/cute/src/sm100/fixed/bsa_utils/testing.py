"""Retained runtime fake-mode check; upstream test helpers are excluded."""
from torch._guards import active_fake_mode


def is_fake_mode() -> bool:
    return active_fake_mode() is not None
