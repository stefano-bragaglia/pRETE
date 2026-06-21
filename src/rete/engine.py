"""Inference engine: the match / resolve / act loop.

:see: Forgy §1.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rete.beta import Instantiation, PNode
from rete.condition import Production
from rete.fact import Fact
from rete.network import ReteNetwork


def _stamp_event(fact: Fact, meta: dict) -> None:
    ts_field = meta.get("timestamp_field")
    if ts_field:
        fact.timestamp = getattr(fact.obj, ts_field)


def _keep(entry: Instantiation, production: Production, pre_ids: set) -> bool:
    """Return True if *entry* should be kept in the conflict set.

    Called after a ``no_loop`` rule fires to filter out self-reactivations.

    :param entry: a conflict-set entry to evaluate.
    :param production: the production that just fired.
    :param pre_ids: ``id()`` values of entries for *production* that existed
        before the firing; entries not in this set were added by the RHS.
    """
    if entry.production is not production:
        return True
    return id(entry) in pre_ids


@dataclass
class InferenceEngine:
    """Wraps :class:`ReteNetwork` with a select-and-fire loop.

    The ``strategy`` callable picks one :class:`Instantiation` from the
    conflict set each cycle; the default is :meth:`recency_strategy`.

    :see: Forgy §1.1
    """

    network: ReteNetwork = field(default_factory=ReteNetwork)
    strategy: Callable[[list[Instantiation]], Instantiation] = field(
        default_factory=lambda: InferenceEngine.recency_strategy
    )
    logical_clock: float = 0.0

    @staticmethod
    def fifo_strategy(conflict_set: list[Instantiation]) -> Instantiation:
        """Return the oldest (first-added) instantiation.

        :param conflict_set: the current conflict set
        """
        return conflict_set[0]

    @staticmethod
    def recency_strategy(conflict_set: list[Instantiation]) -> Instantiation:
        """Return the most recently added instantiation.

        :param conflict_set: the current conflict set
        """
        return conflict_set[-1]

    # ------------------------------------------------------------------
    # Passthrough delegates
    # ------------------------------------------------------------------

    def advance_clock(self, t: float) -> None:
        """Set the logical clock to an absolute value *t*.

        To advance by 30 s: ``engine.advance_clock(engine.logical_clock + 30)``.

        :param t: new absolute clock value (not a delta).
        """
        self.logical_clock = t

    def add_fact(self, fact: Fact) -> None:
        """Assert *fact* into the network, stamping its timestamp for event types.

        :param fact: the :class:`Fact` to add
        """
        meta = getattr(type(fact.obj), "__prl_meta__", None)
        if meta:
            _stamp_event(fact, meta)
        self.network.add_fact(fact)

    def remove_fact(self, fact: Fact) -> None:
        """Retract *fact* from the network.

        :param fact: the :class:`Fact` to remove
        """
        self.network.remove_fact(fact)

    def update_fact(self, fact: Fact) -> None:
        """Retract and re-assert *fact* after its wrapped object has been mutated.

        Equivalent to Drools ``modify``.  Mutate ``fact.obj`` attributes in
        place before calling; do not replace ``fact.obj`` itself, as that
        breaks the back-pointer chain.

        :param fact: the :class:`Fact` whose ``obj`` has been mutated in place
        """
        self.remove_fact(fact)
        self.add_fact(fact)

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
        self._expire_events()
        cs = self.network.conflict_set
        steps = 0
        while cs and (max_steps is None or steps < max_steps):
            inst = self.strategy(cs)
            cs.remove(inst)
            self._fire(inst, cs)
            steps += 1
        return steps

    def _expire_events(self) -> None:
        stale = [f for f in list(self.network.root._facts) if self._is_stale(f)]
        for f in stale:
            self.remove_fact(f)

    def _is_stale(self, fact: Fact) -> bool:
        meta = getattr(type(fact.obj), "__prl_meta__", None)
        if not meta:
            return False
        expires = meta.get("expires_delta")
        if expires is None:
            return False
        if fact.timestamp is None:
            return False
        return fact.timestamp + expires < self.logical_clock

    def _fire(self, inst: Instantiation, cs: list[Instantiation]) -> None:
        """Execute *inst*; suppress self-reactivations for ``no_loop`` rules.

        :param inst: the instantiation being fired.
        :param cs: the live conflict set (mutated in place when no_loop is set).
        """
        if inst.production.no_loop:
            self._fire_no_loop(inst, cs)
        else:
            inst.production.rhs(inst.token)

    def _fire_no_loop(
        self, inst: Instantiation, cs: list[Instantiation]
    ) -> None:
        """Fire *inst* and remove any self-reactivations it adds to *cs*.

        :param inst: the no-loop instantiation being fired.
        :param cs: the live conflict set (mutated in place).
        """
        pre_ids = {id(i) for i in cs if i.production is inst.production}
        inst.production.rhs(inst.token)
        cs[:] = [i for i in cs if _keep(i, inst.production, pre_ids)]
