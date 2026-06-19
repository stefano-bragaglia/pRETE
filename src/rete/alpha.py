"""Alpha network for the Rete algorithm: intra-element tests.

:see: Forgy §2.2.1, Doorenbos §2.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rete.condition import Condition
from rete.fact import WME

if TYPE_CHECKING:
    from rete.beta import RightNode


@dataclass
class AlphaMemory:
    """Stores WMEs that have passed all constant tests for one condition.

    Each join node holds a pointer to one alpha memory as its right input.

    :see: Doorenbos §2.2
    """

    items: list[WME] = field(default_factory=list)
    successors: list[RightNode] = field(default_factory=list, repr=False)

    def activate(self, wme: WME) -> None:
        """Store *wme* and register this memory on the WME's back-pointer list.

        :param wme: the WME that passed all constant tests
        """
        self.items.append(wme)
        wme.alpha_memories.append(self)
        for s in self.successors:
            s.right_activate(wme)

    def deactivate(self, wme: WME) -> None:
        """Remove *wme* from this memory.

        Notifies successors before removal, per Doorenbos §2.5 ordering.

        :param wme: the WME being retracted
        """
        for s in self.successors:
            s.right_retract(wme)
        self.items.remove(wme)
        wme.alpha_memories.remove(self)


@dataclass
class AlphaNode:
    """Tests one constant field of a WME and propagates to children / memory.

    Chains of alpha nodes implement all constant tests for a single condition.
    Nodes with identical ``(field, symbol)`` pairs are shared across conditions.

    :see: Doorenbos §2.2
    """

    field: str
    symbol: str
    children: list[AlphaNode] = field(default_factory=list, repr=False)
    output_memory: AlphaMemory | None = field(default=None, repr=False)

    @staticmethod
    def build_or_share(
        parent: RootNode | AlphaNode,
        field: str,
        symbol: str,
    ) -> AlphaNode:
        """Return an existing child of *parent* with ``(field, symbol)``, or create one.

        :param parent: the node whose children list is searched
        :param field: ``'id'``, ``'attribute'``, or ``'value'``
        :param symbol: the constant string to match
        :returns: a shared or newly-created :class:`AlphaNode`
        :see: Doorenbos §2.2
        """
        for child in parent.children:
            if child.field == field and child.symbol == symbol:
                return child
        node = AlphaNode(field=field, symbol=symbol)
        parent.children.append(node)
        return node

    def activate(self, wme: WME) -> None:
        """Propagate *wme* if it matches this node's constant test.

        :param wme: the WME entering the alpha network
        """
        if getattr(wme, self.field) != self.symbol:
            return
        for child in self.children:
            child.activate(wme)
        if self.output_memory is not None:
            self.output_memory.activate(wme)


@dataclass
class RootNode:
    """Entry point of the alpha network; fans out to all top-level alpha nodes.

    Has no test of its own — every WME passes through.

    :see: Doorenbos §2.2
    """

    children: list[AlphaNode] = field(default_factory=list)
    # ponytail: all-wildcard conditions attach their AlphaMemory here directly
    _wildcard_memory: AlphaMemory | None = field(default=None, repr=False)
    _wmes: list[WME] = field(default_factory=list, repr=False)

    def activate(self, wme: WME) -> None:
        """Dispatch *wme* to all child alpha nodes and the wildcard memory.

        :param wme: the WME entering the alpha network
        """
        self._wmes.append(wme)
        for child in self.children:
            child.activate(wme)
        if self._wildcard_memory is not None:
            self._wildcard_memory.activate(wme)

    def deactivate(self, wme: WME) -> None:
        """Remove *wme* from the global WME store.

        :param wme: the WME being retracted
        """
        self._wmes.remove(wme)

    def _replay_into(self, am: AlphaMemory, constants: list[tuple[str, str]]) -> None:
        """Feed any stored WMEs that satisfy *constants* into *am*.

        Called only when a new :class:`AlphaMemory` is created, so WMEs added
        before the production's alpha memory existed are not lost.

        :param am: the freshly created alpha memory to seed
        :param constants: list of ``(field, symbol)`` pairs that must all match
        """
        for wme in self._wmes:
            if all(getattr(wme, f) == s for f, s in constants):
                am.activate(wme)

    def build_or_share_alpha_memory(self, condition: Condition) -> AlphaMemory:
        """Build or retrieve the :class:`AlphaMemory` for *condition*'s constant tests.

        Walks the three fields in order, adding alpha nodes only for constant
        tests.  Wildcards and ``?``-variables add no node.  If every field is
        non-constant the memory is attached to this root directly.

        :param condition: the condition whose constant fields are compiled
        :returns: the :class:`AlphaMemory` at the end of the chain
        :see: Doorenbos §2.2
        """
        constants = condition.alpha_tests()

        if not constants:
            # ponytail: all-wildcard/variable — memory lives directly on root
            if self._wildcard_memory is None:
                self._wildcard_memory = AlphaMemory()
                self._replay_into(self._wildcard_memory, constants)
            return self._wildcard_memory

        current: RootNode | AlphaNode = self
        for f, s in constants:
            current = AlphaNode.build_or_share(current, f, s)

        if current.output_memory is None:
            current.output_memory = AlphaMemory()
            self._replay_into(current.output_memory, constants)
        return current.output_memory


