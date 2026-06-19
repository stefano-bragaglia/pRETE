"""Alpha network for the Rete algorithm: intra-element, type-indexed dispatch.

Facts are dispatched by walking ``type(fact.obj).__mro__``; each
:class:`AlphaMemory` self-filters via its predicate.  Sharing is keyed by
:meth:`~rete.condition.Pattern.alpha_key` so two patterns that reference the
same function objects share one memory.

.. note::
    Registering an alpha memory under every ancestor in the MRO means a
    ``Cat`` fact may dispatch to a ``Pattern(type_=Dog)`` memory (both
    share ``Animal`` as an ancestor).  The predicate — ``isinstance(fact.obj,
    Dog)`` — correctly rejects it.  Do **not** add an extra isinstance guard
    in :meth:`RootNode.activate`; it would break subclass matching.

:see: Doorenbos §2.2, UPDATE_PLAN §Step 3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from rete.condition import Pattern
from rete.fact import Fact

if TYPE_CHECKING:
    from rete.beta import RightNode


@dataclass
class AlphaMemory:
    """Stores Facts that have passed a pattern's type check and alpha tests.

    Each join node holds a pointer to one alpha memory as its right input.

    :param type_: the object type this memory is for
    :param predicate: ``pattern.matches`` — applied on every activation attempt
    :see: Doorenbos §2.2
    """

    type_: type
    predicate: Callable[[Fact], bool]
    items: list[Fact] = field(default_factory=list)
    successors: list[RightNode] = field(default_factory=list, repr=False)

    def activate(self, fact: Fact) -> None:
        """Store *fact* if it passes the predicate; notify successors.

        :param fact: the working-memory element to test
        """
        if not self.predicate(fact):
            return
        self.items.append(fact)
        fact.alpha_memories.append(self)
        for s in self.successors:
            s.right_activate(fact)

    def deactivate(self, fact: Fact) -> None:
        """Remove *fact* from this memory; notify successors before removal.

        :param fact: the working-memory element being retracted
        """
        for s in self.successors:
            s.right_retract(fact)
        self.items.remove(fact)
        fact.alpha_memories.remove(self)


@dataclass
class RootNode:
    """Entry point of the alpha network; dispatches Facts by MRO type index.

    :see: Doorenbos §2.2, UPDATE_PLAN §Step 3
    """

    # ponytail: MRO registration puts every AM under object; all facts scan
    # all AMs registered for object. Acceptable for shallow hierarchies; move to
    # per-type registration only (no MRO walk) if throughput shows this as a hot path.
    _type_index: dict[type, list[AlphaMemory]] = field(default_factory=dict)
    _key_index: dict[tuple[Any, ...], AlphaMemory] = field(default_factory=dict)
    _facts: list[Fact] = field(default_factory=list, repr=False)

    def activate(self, fact: Fact) -> None:
        """Append *fact* to the global store and dispatch by MRO.

        :param fact: the working-memory element entering the network
        """
        self._facts.append(fact)
        for t in type(fact.obj).__mro__:
            for am in self._type_index.get(t, []):
                am.activate(fact)

    def deactivate(self, fact: Fact) -> None:
        """Remove *fact* from the global store.

        Per-memory removal is driven by back-pointers on ``fact.alpha_memories``
        and handled by the caller (see ``ReteNetwork.remove_fact``).

        :param fact: the working-memory element being retracted
        """
        self._facts.remove(fact)

    def build_or_share_alpha_memory(self, pattern: Pattern) -> AlphaMemory:
        """Return the :class:`AlphaMemory` for *pattern*, creating it if needed.

        Sharing: two patterns with the same :meth:`~Pattern.alpha_key` (same
        ``type_`` and identical function objects in ``alpha_tests``) reuse one
        memory.  On creation, existing facts are replayed through the new memory
        so that facts added before this production's alpha memory existed are
        not lost.

        :param pattern: the pattern whose alpha memory is needed
        :returns: an existing or newly-created :class:`AlphaMemory`
        :see: Doorenbos §2.2
        """
        key = pattern.alpha_key()
        if key in self._key_index:
            return self._key_index[key]
        am = AlphaMemory(type_=pattern.type_, predicate=pattern.matches)
        self._key_index[key] = am
        # Register under pattern.type_ only; activate() walks fact.__mro__ for dispatch.
        # Registering under the full MRO would cause double-activation when a fact's own
        # MRO overlaps with the registration set (e.g. Block registered under both Block
        # and object; a Block fact walks (Block, object) and finds the AM twice).
        self._type_index.setdefault(pattern.type_, []).append(am)
        for fact in self._facts:
            am.activate(fact)
        return am
