# pRETE (Python RETE)

A pure-Python implementation of the Rete algorithm for production rule systems.

![pRETE logo](images/pRETE-logo-small.png)

## Background

Implements the algorithm from:
- Forgy, C. L. (1982). Rete: A fast algorithm for the many pattern/many object pattern match problem. *Artificial Intelligence*, 19(1), 17–37.
- Doorenbos, R. B. (1995). *Production system techniques for large rule bases* (CMU-CS-95-113). Carnegie Mellon University.

Working memory elements are represented as `(id, attribute, value)` triples per Doorenbos §2.1.

## Install

```bash
pip install -e .[dev]
```

## Usage

```python
from rete.network import ReteNetwork

net = ReteNetwork()
# add productions and WMEs — see tests/ for examples
```

## Dev

```bash
xenon --max-absolute A --max-modules A --max-average A src/ tests/
ruff check src/ tests/
pytest --cov
```
