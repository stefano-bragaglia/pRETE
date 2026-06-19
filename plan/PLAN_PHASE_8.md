# Phase 8 — Conjunctive Negations

Sources: Doorenbos §2.8.

---

## Context

Phase 7 added `NegativeJoinNode` (NJN) for single negated conditions.  Phase 8
adds **NCC** (Negated Conjunctive Condition): a group of conditions that must
*collectively* have no match for the production to fire.

An NCC requires two cooperating nodes:

- **NccPartnerNode** — a p-node-like sink at the end of the negated
  subnetwork.  It forwards each arriving result token to the NCC node.
- **NccNode** — sits on the main join chain.  Like NJN it uses a count per
  left token, but results come from the partner rather than directly from an
  alpha memory.  It propagates the left token downstream only while its count
  is zero.

The extra complexity versus NJN: partner tokens may arrive *before* the NCC
node has created the `NccToken` for their owner (this happens during
`add_production` initialisation when the subnetwork is seeded first).  The NCC
node buffers those early results in `new_result_buffer` and drains them when
the corresponding left activation arrives.

---

## Pre-existing bug — beta_tokens double-retraction (fix here)

`BetaMemory.left_activate` and `PNode.left_activate` both append
`(token, self)` to `token.wmes[-1].beta_tokens`.  When a NJN or NCC node
propagates a token downstream **without extending it** (same `wmes` tuple), the
same last WME ends up with two entries in `beta_tokens` — one for the upstream
`BetaMemory` and one for the downstream `BetaMemory` / `PNode`.

`JoinNode.right_retract` iterates `wme.beta_tokens` (snapshot) and calls
`mem.left_retract(token)` for each entry.  The first call cascades through the
upstream BetaMemory → NJN / NccNode → downstream node and removes the token.
The second call then finds the token already gone and raises `ValueError`.

The bug is latent in Phase 7 code but no existing test removes the *positive*
WME of a `[pos_cond, neg_cond]` production while the match is live (all
Phase 7 removal tests remove only the blocking WME).  It surfaces as soon as
NCC propagates tokens into a downstream BetaMemory.

**Fix**: add an idempotency guard to both `BetaMemory.left_retract` and
`PNode.left_retract`:

```python
if token not in self.items:
    return
```

Add one regression test in `test_beta.py` covering the NJN double-retraction
path, and one in `test_network.py` covering the positive-WME removal case.

---

## Files to create / modify

```
src/rete/condition.py   add NccGroup
src/rete/beta.py        add NccToken, NccPartnerNode, NccNode;
                        fix BetaMemory.left_retract / PNode.left_retract;
                        update parent_join / lhs type annotations
src/rete/network.py     handle NccGroup in _build_join_chain;
                        add _build_ncc, _build_ncc_subnetwork,
                        _process_condition, _gc_ncc_node;
                        update _gc_beta_memory, remove_production,
                        _build_or_share_beta_memory type annotations
tests/test_beta.py      bug regression + NccToken / NccPartnerNode /
                        NccNode unit tests
tests/test_network.py   NCC integration tests + positive-WME regression
```

---

## Objects

### `condition.py`

#### `NccGroup`

```python
@dataclass(frozen=True)
class NccGroup:
    """A group of conditions to be conjunctively negated (Doorenbos §2.8)."""
    conditions: tuple[Condition, ...]
```

`Production.lhs` type widens from `list[Condition]` to
`list[Condition | NccGroup]`.

---

### `beta.py`

#### `NccToken`

```python
@dataclass
class NccToken:
    """Left token paired with its count of subnetwork matches and their results."""
    token: Token
    count: int = 0
    results: list[Token] = field(default_factory=list)
```

`results` holds the partner tokens that belong to this owner — needed for
direct cleanup in `NccNode.left_retract` without a full partner scan.

---

#### `NccPartnerNode`

Sink at the end of the NCC subnetwork.  Stores received result tokens in
`items` (for retraction scans) and delegates count bookkeeping to `ncc_node`.
Also stores `sub_last_join` for GC (set by `_build_ncc`).

Fields:
```
items: list[Token]
ncc_node: NccNode | None
sub_last_join: JoinNode | None    # set during wiring, not __init__
```

**`left_activate(token)`** — complexity 4

```
append to items
owner_wmes = token.wmes[:ncc_node.owner_length]
ncc_token  = _find_ncc_token(owner_wmes)       # None if not yet created
if ncc_token:
    append token to ncc_token.results
    if ncc_token.count == 0:
        for child in ncc_node.children: child.left_retract(ncc_token.token)
    ncc_token.count += 1
else:
    ncc_node.new_result_buffer.append(token)
```

Retract happens *before* increment (same invariant as NJN).

**`left_retract(token)`** — complexity 4

```
items.remove(token)
owner_wmes = token.wmes[:ncc_node.owner_length]
ncc_token  = _find_ncc_token(owner_wmes)
if ncc_token:
    ncc_token.results.remove(token)
    ncc_token.count -= 1
    if ncc_token.count == 0:
        for child in ncc_node.children: child.left_activate(ncc_token.token)
else:
    ncc_node.new_result_buffer.remove(token)
```

Decrement happens *before* zero-check (same invariant as NJN).

**`_find_ncc_token(owner_wmes) -> NccToken | None`** — complexity 3

```python
return next(
    (nt for nt in self.ncc_node.items if nt.token.wmes == owner_wmes),
    None,
)
```

---

#### `NccNode`

Consumer on the main join chain.  Registered in `left_input.successors` (when
left_input is a `BetaMemory`) so it receives `left_activate` / `left_retract`
from the normal beta-propagation mechanism.

Fields:
```
children: list
partner: NccPartnerNode | None
items: list[NccToken]
new_result_buffer: list[Token]
owner_length: int               # len(earlier) at build time
left_input: BetaMemory | DummyTopNode
```

**`left_activate(token)`** — complexity 3

```
results  = _drain_buffer(token)
ncc_tok  = NccToken(token=token, count=len(results), results=results)
items.append(ncc_tok)
if ncc_tok.count == 0:
    for child in children: child.left_activate(token)
```

**`_drain_buffer(token) -> list[Token]`** — complexity 3

Partition `new_result_buffer` into matching / non-matching; replace buffer
with non-matching; return matching:

```python
keep, drain = [], []
for r in self.new_result_buffer:
    if r.wmes[:self.owner_length] == token.wmes:
        drain.append(r)
    else:
        keep.append(r)
self.new_result_buffer[:] = keep
return drain
```

**`left_retract(token)`** — complexity 3

```
ncc_tok = _find_ncc_token(token)
_retract_partner_results(ncc_tok)
items.remove(ncc_tok)
if ncc_tok.count == 0:
    for child in children: child.left_retract(token)
```

**`_find_ncc_token(token) -> NccToken`** — complexity 3

```python
return next(nt for nt in self.items if nt.token is token)
```

Identity comparison (`is`), not equality.

**`_retract_partner_results(ncc_tok)`** — complexity 2

```python
for result in list(ncc_tok.results):
    self.partner.items.remove(result)
ncc_tok.results.clear()
```

**`update_child(child)`** — complexity 3

```python
for ncc_tok in self.items:
    if ncc_tok.count == 0:
        child.left_activate(ncc_tok.token)
```

---

#### Updated type annotations in existing classes

| Class | Field | Old type | New type |
|---|---|---|---|
| `BetaMemory` | `parent_join` | `JoinNode \| NegativeJoinNode \| None` | `JoinNode \| NegativeJoinNode \| NccNode \| None` |
| `PNode` | `parent_join` | `JoinNode \| NegativeJoinNode \| None` | `JoinNode \| NegativeJoinNode \| NccNode \| None` |

No logic changes beyond the idempotency guard.

---

### `network.py`

#### `_build_join_chain` — complexity 4 (unchanged)

Replace the `if cond.negated` branch with an `isinstance(item, NccGroup)` branch:

```python
for i, item in enumerate(lhs):
    if isinstance(item, NccGroup):
        last = self._build_ncc(left, earlier, item)
        # NCC does not extend 'earlier' — variables inside the subnetwork
        # are not referenceable by subsequent main-chain conditions
    else:
        last = self._process_condition(item, left, earlier)
        earlier.append(item)
    if i < len(lhs) - 1:
        left = self._build_or_share_beta_memory(last)
```

#### `_process_condition(cond, left, earlier) -> JoinNode | NegativeJoinNode` — complexity 2

Extracted helper (keeps `_build_join_chain` within budget):

```python
am    = self.root.build_or_share_alpha_memory(cond)
tests = JoinTest.extract(cond, earlier)
if cond.negated:
    return self._build_or_share_negative_join_node(left, am, tests)
return self._build_or_share_join_node(left, am, tests)
```

#### `_build_ncc(left, earlier, group) -> NccNode` — complexity 3

```python
ncc_node = NccNode(owner_length=len(earlier), left_input=left)
partner  = NccPartnerNode(ncc_node=ncc_node)
ncc_node.partner = partner
sub_last = self._build_ncc_subnetwork(left, earlier, group)
sub_last.children.append(partner)
partner.sub_last_join = sub_last
if isinstance(left, BetaMemory):
    left.successors.append(ncc_node)
sub_last.update_child(partner)       # seeds new_result_buffer
for token in left.items:             # drains buffer while creating NccTokens
    ncc_node.left_activate(token)
return ncc_node
```

Ordering: partner seeded first so results land in buffer; NCC node then drains.

#### `_build_ncc_subnetwork(left, earlier, group) -> JoinNode` — complexity 3

```python
sub_left, sub_earlier, sub_last = left, list(earlier), None
for i, cond in enumerate(group.conditions):
    am    = self.root.build_or_share_alpha_memory(cond)
    tests = JoinTest.extract(cond, sub_earlier)
    sub_last = self._build_or_share_join_node(sub_left, am, tests)
    sub_earlier.append(cond)
    if i < len(group.conditions) - 1:
        sub_left = self._build_or_share_beta_memory(sub_last)
return sub_last
```

Subnetwork only uses positive `JoinNode`s (NCC conditions are implicitly
positive; their *collective* absence is what the NCC node tests).

#### `_build_or_share_beta_memory` — type annotation only

Widen `parent_join` parameter to `JoinNode | NegativeJoinNode | NccNode`.
No body changes; all three types expose `.children` and `.update_child`.

#### `_gc_ncc_node(ncc: NccNode)` — complexity 4

```python
if ncc.children:
    return
if isinstance(ncc.left_input, BetaMemory):
    ncc.left_input.successors.remove(ncc)
    self._gc_beta_memory(ncc.left_input)
partner = ncc.partner
if partner and partner.sub_last_join:
    partner.sub_last_join.children.remove(partner)
    self._gc_join_node(partner.sub_last_join)
```

#### `_gc_beta_memory` — add `NccNode` dispatch, complexity 4

```python
if bm.successors or bm.parent_join is None:
    return
bm.parent_join.children.remove(bm)
if isinstance(bm.parent_join, NegativeJoinNode):
    self._gc_negative_join_node(bm.parent_join)
elif isinstance(bm.parent_join, NccNode):
    self._gc_ncc_node(bm.parent_join)
else:
    self._gc_join_node(bm.parent_join)
```

#### `remove_production` — add `NccNode` dispatch, complexity 4

```python
for token in list(p_node.items):
    p_node.left_retract(token)
jn = p_node.parent_join
jn.children.remove(p_node)
p_node.parent_join = None
if isinstance(jn, NegativeJoinNode):
    self._gc_negative_join_node(jn)
elif isinstance(jn, NccNode):
    self._gc_ncc_node(jn)
else:
    self._gc_join_node(jn)
```

---

## Tests

### `tests/test_beta.py` — appended

#### Bug regression (double-retraction via NJN)

| Test | What it checks |
|---|---|
| `test_beta_memory_left_retract_idempotent` | calling `left_retract` twice on the same token does not raise |
| `test_pnode_left_retract_idempotent` | same for PNode |

#### `NccToken`

| Test | What it checks |
|---|---|
| `test_ncc_token_defaults` | `count == 0`, `results == []` |

#### `NccPartnerNode.left_activate`

| Test | What it checks |
|---|---|
| `test_ncc_partner_activate_buffers_when_no_ncc_token` | no matching NccToken → result appended to `new_result_buffer` |
| `test_ncc_partner_activate_increments_count` | matching NccToken present (count 0) → count becomes 1 |
| `test_ncc_partner_activate_retracts_when_count_was_zero` | count 0 → child.left_retract called before increment |
| `test_ncc_partner_activate_no_retract_when_already_blocked` | count 1 → no retract, count becomes 2 |

#### `NccPartnerNode.left_retract`

| Test | What it checks |
|---|---|
| `test_ncc_partner_retract_from_buffer` | result in buffer → removed from buffer, no count change |
| `test_ncc_partner_retract_decrements_count` | count 1 → 0, result removed from ncc_token.results |
| `test_ncc_partner_retract_asserts_when_count_reaches_zero` | count 1 → 0 → child.left_activate called |
| `test_ncc_partner_retract_no_assert_count_stays_positive` | count 2 → 1 → no assert |

#### `NccNode.left_activate`

| Test | What it checks |
|---|---|
| `test_ncc_node_activate_empty_buffer_propagates` | buffer empty → NccToken count 0 → child receives token |
| `test_ncc_node_activate_drains_matching_buffer` | 1 matching result in buffer → count 1 → no propagation, buffer empty |
| `test_ncc_node_activate_ignores_nonmatching_buffer` | 1 non-matching result in buffer → count 0 → child receives token, buffer unchanged |

#### `NccNode.left_retract`

| Test | What it checks |
|---|---|
| `test_ncc_node_retract_propagated_token` | count 0 → child.left_retract called |
| `test_ncc_node_retract_blocked_token` | count > 0 → child not called |
| `test_ncc_node_retract_clears_partner_items` | partner.items emptied for this token |

#### `NccNode.update_child`

| Test | What it checks |
|---|---|
| `test_ncc_node_update_child_only_propagated` | count-0 NccToken sent to child; count-1 skipped |

---

### `tests/test_network.py` — appended

#### Bug regression

| Test | What it checks |
|---|---|
| `test_remove_positive_wme_with_active_match` | `[pos_cond, neg_cond]`, match live, remove pos_wme → no error, conflict set empty |

#### NCC integration

| Test | What it checks |
|---|---|
| `test_ncc_no_match_fires` | `NccGroup` with one cond; no WME → fires |
| `test_ncc_blocked_by_match` | subnetwork match added → conflict set empty |
| `test_ncc_unblocked_on_retraction` | match removed → fires again |
| `test_ncc_two_conditions_both_absent_fires` | two-condition NccGroup, both absent → fires |
| `test_ncc_two_conditions_one_present_blocked` | one of two NCC conditions matched → blocked |
| `test_ncc_positive_then_ncc_fires` | `[pos_cond, NccGroup([neg_cond])]`; pos present, NCC absent → fires |
| `test_ncc_positive_then_ncc_blocked` | both pos and NCC match present → blocked |
| `test_ncc_positive_then_ncc_unblocked` | pos present; NCC match added then removed → fires |
| `test_ncc_retroactive_retraction` | production fires (count 0), then a subnetwork match arrives → retracted |
| `test_ncc_initialization_with_existing_wmes` | WMEs present before `add_production`; NCC match exists → not fired (buffer drained correctly) |
| `test_ncc_gc_on_remove_production` | `add_production` with NCC then `remove_production` → no error, conflict set empty |

---

## Criticalities

**1. Initialisation order — partner seeded before NCC node**

`_build_ncc` must call `sub_last.update_child(partner)` *before* looping over
`left.items` to activate the NCC node.  Reversing the order means results
arrive after NccTokens exist, so `_find_ncc_token` succeeds immediately and
the `new_result_buffer` is never used — but then the NCC node may over-count
if `left.items` was already populated before the partner.  Keep the order:
seed partner → drain buffer via NCC left activations.

**2. Retroactive retraction (the hardest case)**

If the NCC node has already propagated a token downstream (count was 0) and a
partner result then arrives, `NccPartnerNode.left_activate` must retract before
incrementing count.  The guard `if ncc_token.count == 0: retract` must come
before `ncc_token.count += 1`.

**3. `_drain_buffer` must compare `wmes[:owner_length]`**

Partner results are extended tokens (length = `owner_length + k`).  Matching
requires comparing only the first `owner_length` WMEs of the result against
the full `token.wmes`.

**4. Identity vs. equality in `_find_ncc_token(token)`**

`nt.token is token` not `nt.token == token`.  Two `Token` objects with
identical `wmes` compare equal under the default dataclass `__eq__`.

**5. Pre-existing double-retraction bug**

Fix `BetaMemory.left_retract` and `PNode.left_retract` with the idempotency
guard before implementing NCC; the NCC node propagates unextended tokens,
which immediately exercises the bug path.  Add the guard and the regression
tests first, confirm they pass, then proceed.

**6. `earlier` does not grow after an NccGroup**

Variables inside the NCC subnetwork are not accessible in subsequent
main-chain conditions.  `_build_join_chain` must NOT call `earlier.append(item)`
when `item` is an `NccGroup`.

**7. Subnetwork uses only positive JoinNodes**

The conditions inside `NccGroup.conditions` should not have `negated=True`.
`_build_ncc_subnetwork` uses `_build_or_share_join_node` unconditionally.
Add a runtime `assert not cond.negated` guard inside the loop if defensive
checks are desired.

**8. `NccNode` in `left_input.successors`**

`NccNode` receives its left activations through the normal
`BetaMemory.left_activate` → `s.left_activate(token)` loop.  The NCC node
must be appended to `left_input.successors` in `_build_ncc` (when
`left_input` is a `BetaMemory`); if it is a `DummyTopNode`, seeding is done
via the explicit `for token in left.items` loop at build time.

**9. GC stores `sub_last_join` on the partner**

`_gc_ncc_node` must know which join node the partner was attached to in order
to unlink and GC the subnetwork.  Set `partner.sub_last_join = sub_last` in
`_build_ncc` immediately after `sub_last.children.append(partner)`.

**10. Complexity budget**

| Method | Complexity |
|---|---|
| `NccPartnerNode.left_activate` | 4 |
| `NccPartnerNode._find_ncc_token` | 3 |
| `NccPartnerNode.left_retract` | 4 |
| `NccNode.left_activate` | 3 |
| `NccNode._drain_buffer` | 3 |
| `NccNode.left_retract` | 3 |
| `NccNode._find_ncc_token` | 3 |
| `NccNode._retract_partner_results` | 2 |
| `NccNode.update_child` | 3 |
| `ReteNetwork._build_join_chain` (updated) | 4 |
| `ReteNetwork._process_condition` | 2 |
| `ReteNetwork._build_ncc` | 3 |
| `ReteNetwork._build_ncc_subnetwork` | 3 |
| `ReteNetwork._gc_ncc_node` | 4 |
| `ReteNetwork._gc_beta_memory` (updated) | 4 |
| `ReteNetwork.remove_production` (updated) | 4 |

Run `hatch run check` before every commit.

---

## Implementation notes — non-obvious design calls discovered during coding

These were not predicted in the plan above but emerged during the red→green
cycle.  Recorded here so future phases can refer to the reasoning rather than
rediscover it.

---

### A. `NccPartnerNode` must register in `wme.beta_tokens`

**What the plan said**: the partner "forwards each arriving result token to the
NCC node".

**What was missing**: the partner is the *direct child* of the last subnetwork
join node — not a `BetaMemory`.  `JoinNode.right_retract` retracts tokens
exclusively by iterating `wme.beta_tokens`.  `BetaMemory.left_activate`
registers there; the partner does not — so when a subnetwork WME is removed,
`right_retract` finds nothing and the NCC count is never decremented.  Matches
remain permanently blocked.

**Fix**: `NccPartnerNode.left_activate` appends `(token, self)` to
`token.wmes[-1].beta_tokens` (mirroring `BetaMemory.left_activate`).
`NccPartnerNode.left_retract` removes it.

**Why it didn't surface as a plan issue**: the plan described the partner only
in terms of its role toward the NCC node.  The WME retraction path through
`JoinNode.right_retract → wme.beta_tokens` was only visible by tracing what
happens at runtime when `remove_wme` is called on a subnetwork WME.

**Complexity impact**: `left_activate` goes from complexity 4 to 5 (adds one
`if token.wmes` branch); `left_retract` same.  Both stay within budget.

---

### B. Successor ordering in `left_input.successors` is load-bearing

**What happens**: `_build_ncc` appends two objects to `left_input.successors`
(when `left_input` is a `BetaMemory`):

1. `sub_first_join` — appended inside `_build_ncc_subnetwork` →
   `_build_or_share_join_node`, which calls `left.successors.append(jn)`.
2. `ncc_node` — appended explicitly at the end of `_build_ncc`.

Because `_build_ncc_subnetwork` is called *before* the explicit append, the
subnetwork join is always first in `left_input.successors`.

**Why the order matters**: when the main chain's BetaMemory fires
`left_retract(token)`, it calls all successors in order.  If `sub_first_join`
fires first, it cascades through the subnetwork to
`partner.left_retract(result)`, which:

- removes `result` from `partner.items`
- removes `result` from `ncc_tok.results`
- decrements `ncc_tok.count`

By the time `ncc_node.left_retract(token)` fires second,
`_retract_partner_results` iterates `ncc_tok.results` — which is already empty.
No double-removal, no crash.

If the order were reversed (`ncc_node` first), `_retract_partner_results` would
remove results from `partner.items`, then `sub_first_join.left_retract` would
call `partner.left_retract(result)` and fail with `ValueError` because `result`
is already gone.

**The invariant**: *the subnetwork's first join node must appear in
`left_input.successors` before the NccNode.*  `_build_ncc`'s current
structure preserves this automatically (subnetwork built before NccNode is
wired), but any refactor that reorders those steps would break it silently.

---

### C. Partner `left_retract` unit tests must go through `left_activate`

**What happened**: the original unit tests for `NccPartnerNode.left_retract`
manually populated `partner.items` and `ncc_tok.results` by direct list
mutation, bypassing `left_activate`.  After design call A was implemented
(partner registers in `wme.beta_tokens`), `left_retract` tried to remove
`(token, self)` from `wme.beta_tokens` — which was never added — and raised
`ValueError`.

**Fix**: the three affected tests were updated to set up state through
`partner.left_activate(result)` instead of direct mutation.  This respects the
invariant: any token in `partner.items` was placed there by `left_activate` and
is therefore registered in `wme.beta_tokens`.

**Rule for future partner unit tests**: never directly append to `partner.items`
or `ncc_tok.results`.  Always call `partner.left_activate`.

---

### D. `test_ncc_initialization_with_existing_wmes` requires a pre-built alpha network

**What happened**: the test called `net.add_wme(wme)` before
`net.add_production(...)`.  Because no alpha nodes existed yet, `root.activate`
routed the WME to nothing and it was silently lost.  `add_production` then
built the alpha network from scratch, found the alpha memory empty, and the NCC
match was never detected.  The production fired incorrectly.

**Fix**: the test calls
`net.root.build_or_share_alpha_memory(Condition(...))` first to create the alpha
network, then `add_wme`, then `add_production`.  This matches the pattern in the
existing `test_integration_activate_wme_then_add_production`.

**General rule**: `add_wme` before `add_production` only works correctly when
the alpha network is already built.  The same constraint applies to any test
that checks initialisation replay with pre-existing WMEs.
