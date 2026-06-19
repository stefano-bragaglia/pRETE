"""Inference engine: the match / resolve / act loop.

:see: Forgy §1.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rete.beta import Instantiation, PNode
from rete.condition import Production
from rete.network import ReteNetwork
from rete.wme import WME


def fifo_strategy(conflict_set: list[Instantiation]) -> Instantiation:
    """Return the oldest (first-added) instantiation.

    :param conflict_set: the current conflict set
    """
    return conflict_set[0]


def recency_strategy(conflict_set: list[Instantiation]) -> Instantiation:
    """Return the most recently added instantiation.

    :param conflict_set: the current conflict set
    """
    return conflict_set[-1]


@dataclass
class InferenceEngine:
    """Wraps :class:`ReteNetwork` with a select-and-fire loop.

    The ``strategy`` callable picks one :class:`Instantiation` from the
    conflict set each cycle; the default is :func:`recency_strategy`.

    :see: Forgy §1.1
    """

    network: ReteNetwork = field(default_factory=ReteNetwork)
    strategy: Callable[[list[Instantiation]], Instantiation] = field(
        default_factory=lambda: recency_strategy
    )

    # ------------------------------------------------------------------
    # Passthrough delegates
    # ------------------------------------------------------------------

    def add_wme(self, wme: WME) -> None:
        """Assert *wme* into the network.

        :param wme: the WME to add
        """
        self.network.add_wme(wme)

    def remove_wme(self, wme: WME) -> None:
        """Retract *wme* from the network.

        :param wme: the WME to remove
        """
        self.network.remove_wme(wme)

    def add_production(self, production: Production) -> PNode:
        """Compile *production* into the network.

        :param production: the production to add
        :returns: the terminal :class:`PNode`
        """
        return self.network.add_production(production)

    def remove_production(self, p_node: PNode) -> None:
        """Unlink *p_node* from the network.

        :param p_node: the :class:`PNode` returned by :meth:`add_production`
        """
        self.network.remove_production(p_node)

    # ------------------------------------------------------------------
    # The loop (Forgy §1.1)
    # ------------------------------------------------------------------

    def run(self, max_steps: int | None = None) -> int:
        """Fire instantiations until the conflict set is empty or *max_steps* is hit.

        The selected instantiation is removed from the conflict set *before*
        its RHS executes so that the RHS cannot re-select the same pair.

        :param max_steps: optional cap on the number of firings
        :returns: number of instantiations fired
        """
        cs = self.network.conflict_set
        steps = 0
        while cs and (max_steps is None or steps < max_steps):
            inst = self.strategy(cs)
            cs.remove(inst)
            inst.production.rhs(inst.token)
            steps += 1
        return steps
