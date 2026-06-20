"""Lexer for pRETE Rule Language (PRL).

Converts a PRL source string into a flat list of :class:`Tok` instances.
The then-block of each rule is captured verbatim as a single RAWBLOCK token
so that arbitrary Python code inside ``then … end`` does not interfere with
the PRL token set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Tok", "tokenize"]

# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------

_KEYWORDS: frozenset[str] = frozenset({
    "package", "declare", "extends", "rule", "end",
    "when", "then", "not", "salience",
    "import", "from", "as",
    "true", "false", "null", "None",
    "True", "False",
})

_SILENT: frozenset[str] = frozenset({"COMMENT_BLOCK", "COMMENT_LINE", "SPACE"})

# Group order is semantic: first match wins.
# NOLOOP before WORD so 'no-loop' is not split.
# FLOAT before INT so '3.14' is not split at the dot.
# Two-char OP alternatives before one-char to avoid '<' eating '<='.
# COMMENT groups before PUNCT so '//' is not two PUNCT('/') tokens.
# AT before PUNCT so '@' is not left unmatched (absent from PUNCT charset).
_MASTER = re.compile(
    r"(?P<COMMENT_BLOCK>/\*.*?\*/)"
    r"|(?P<COMMENT_LINE>//[^\n]*)"
    r"|(?P<NOLOOP>no-loop)"
    r"|(?P<FLOAT>\d+\.\d+)"
    r"|(?P<INT>\d+)"
    r"|(?P<STRING>\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*')"
    r"|(?P<VAR>\$[A-Za-z_]\w*)"
    r"|(?P<OP>==|!=|<=|>=|<|>|=)"
    r"|(?P<WORD>[A-Za-z_]\w*)"
    r"|(?P<AT>@)"
    r"|(?P<PUNCT>[,./()[\]{};:\-])"
    r"|(?P<SPACE>[ \t\r\n]+)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Tok
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tok:
    """A single PRL lexical token.

    :param kind: token category (KW, IDENT, VAR, STRING, FLOAT, INT,
        OP, PUNCT, RAWBLOCK).
    :param value: matched source text, preserved verbatim.
    :param line: 1-based source line number at the point of emission.
    """

    kind: str
    value: str
    line: int


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    """Converts PRL source text into a flat list of :class:`Tok` instances.

    :param text: full PRL source string (any line ending; ``\\r\\n``
        is normalised to ``\\n`` on construction).
    """

    def __init__(self, text: str) -> None:
        self._text: str = text.replace("\r\n", "\n").replace("\r", "\n")
        self._pos: int = 0
        self._line: int = 1
        self._toks: list[Tok] = []

    # -- public ---------------------------------------------------------------

    def tokenize(self) -> list[Tok]:
        """Return all tokens for the source text.

        :returns: ordered list of :class:`Tok` instances.
        """
        while self._pos < len(self._text):
            if self._scan_token() and self._last_is_then():
                self._scan_rawblock()
        return self._toks

    # -- private helpers ------------------------------------------------------

    def _last_is_then(self) -> bool:
        """Return True iff the most recently emitted token is KW("then")."""
        return (
            bool(self._toks)
            and self._toks[-1].kind == "KW"
            and self._toks[-1].value == "then"
        )

    def _emit(self, kind: str, value: str) -> None:
        """Append Tok(kind, value, current_line) to the accumulator."""
        self._toks.append(Tok(kind, value, self._line))

    def _classify(self, kind: str, value: str) -> str:
        """Map a regex group name to a token kind.

        :param kind: name of the matched regex group.
        :param value: matched text (used to distinguish KW from IDENT).
        :returns: token kind string.
        """
        if kind == "NOLOOP":
            return "KW"
        if kind == "WORD":
            return "KW" if value in _KEYWORDS else "IDENT"
        return kind

    def _scan_token(self) -> bool:
        """Match one token at the current position.

        :returns: ``True`` if a :class:`Tok` was emitted, ``False`` if only
            whitespace or a comment was consumed.
        :raises SyntaxError: on an unrecognised character.
        """
        m = _MASTER.match(self._text, self._pos)
        if not m:
            raise SyntaxError(
                f"Unexpected character at line {self._line}: "
                f"{self._text[self._pos]!r}"
            )
        self._pos = m.end()
        kind, value = m.lastgroup, m.group()
        self._line += value.count("\n")
        if kind in _SILENT:
            return False
        self._emit(self._classify(kind, value), value)
        return True

    def _skip_to_next_line(self) -> None:
        """Advance past any remaining characters on the current line."""
        nl = self._text.find("\n", self._pos)
        if nl == -1:
            self._pos = len(self._text)
        else:
            self._pos = nl + 1
            self._line += 1

    def _read_line(self) -> str:
        """Read and return one complete source line (including its newline).

        :returns: the line text; ``self._line`` is incremented when a newline
            is consumed.
        """
        nl = self._text.find("\n", self._pos)
        if nl == -1:
            line = self._text[self._pos:]
            self._pos = len(self._text)
        else:
            line = self._text[self._pos : nl + 1]
            self._pos = nl + 1
            self._line += 1
        return line

    def _scan_rawblock(self) -> None:
        """Capture lines verbatim until a line stripped to exactly ``end``.

        Emits one RAWBLOCK token (the accumulated Python source, whose
        ``line`` attribute is the line of the ``then`` keyword) followed by
        one KW(``end``) token.

        :raises SyntaxError: if end-of-file is reached before ``end`` is found.
        """
        start_line = self._line
        self._skip_to_next_line()
        lines: list[str] = []
        while self._pos < len(self._text):
            end_line = self._line
            line = self._read_line()
            if line.strip() == "end":
                self._toks.append(Tok("RAWBLOCK", "".join(lines), start_line))
                self._toks.append(Tok("KW", "end", end_line))
                return
            lines.append(line)
        raise SyntaxError(
            f"Unterminated then-block opening at line {start_line}"
        )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[Tok]:
    """Tokenize PRL source text into a flat token list.

    :param text: PRL source string (any line ending).
    :returns: ordered list of :class:`Tok` instances.
    """
    return Lexer(text).tokenize()
