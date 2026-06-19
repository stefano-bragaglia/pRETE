"""Working memory elements and partial-match tokens for the Rete algorithm.

:see: Doorenbos §2.1, §2.3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rete.alpha import AlphaMemory
    from rete.beta import LeftNode


@dataclass(eq=False)
class Fact:
    """A working memory element wrapping an arbitrary Python object.

    Hashed and compared by identity so two Facts wrapping equal objects
    remain distinct entries in working memory.

    :see: Doorenbos §2.1
    """

    obj: Any
    alpha_memories: list[AlphaMemory] = field(default_factory=list, repr=False)
    beta_tokens: list[tuple[Token, LeftNode]] = field(default_factory=list, repr=False)


# ponytail: temporary alias; removed in Step 7 when __init__.py is cleaned up
WME = Fact


@dataclass
class Token:
    """An ordered sequence of Facts plus named variable bindings for a partial match.

    Extend immutably:
    ``Token(facts=parent.facts + (fact,), bindings={**parent.bindings, **new})``.

    :see: Doorenbos §2.3
    """

    # ponytail: flat tuple; switch to linked-list (parent + fact) if memory grows.
    facts: tuple[Fact, ...] = ()
    bindings: dict[str, Any] = field(default_factory=dict)
