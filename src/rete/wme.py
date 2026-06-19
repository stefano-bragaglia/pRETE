from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(eq=False)
class WME:
    """Working memory element: a (id, attribute, value) triple (Doorenbos §2.1)."""

    id: str
    attribute: str
    value: str
    # ponytail: untyped lists; typed to AlphaMemory/Token once those exist (Phase 2/6)
    alpha_memories: list = field(default_factory=list, repr=False)
    beta_tokens: list = field(default_factory=list, repr=False)


@dataclass
class Token:
    """Ordered partial match (Doorenbos §2.3).

    Extend by constructing a new token: Token(wmes=parent.wmes + (wme,)).
    ponytail: flat tuple; switch to linked-list (parent + wme) if memory grows.
    """

    wmes: tuple[WME, ...] = ()
