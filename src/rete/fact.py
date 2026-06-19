"""Working memory elements and partial-match tokens for the Rete algorithm.

:see: Doorenbos §2.1, §2.3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rete.alpha import AlphaMemory
    from rete.beta import LeftNode


@dataclass(eq=False)
class WME:
    """A working memory element: an ``(id, attribute, value)`` triple.

    Hashed and compared by identity so two WMEs with identical fields
    remain distinct facts in working memory.

    :see: Doorenbos §2.1
    """

    id: str
    attribute: str
    value: str
    alpha_memories: list[AlphaMemory] = field(default_factory=list, repr=False)
    beta_tokens: list[tuple[Token, LeftNode]] = field(default_factory=list, repr=False)


@dataclass
class Token:
    """An ordered sequence of WMEs representing a partial match.

    Extend immutably: ``Token(wmes=parent.wmes + (wme,))``.

    :see: Doorenbos §2.3
    """

    # ponytail: flat tuple; switch to linked-list (parent + wme) if memory grows.
    wmes: tuple[WME, ...] = ()
