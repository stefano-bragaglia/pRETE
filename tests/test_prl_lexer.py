"""Unit tests for the PRL lexer (``src/rete/prl_lexer.py``).

Tests are committed before the implementation and are expected to fail
with an ``ImportError`` until ``prl_lexer.py`` is created.
"""
from __future__ import annotations

import pytest

from rete.prl_lexer import Tok, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kv(text: str) -> list[tuple[str, str]]:
    """Return (kind, value) pairs for every token in *text*."""
    return [(t.kind, t.value) for t in tokenize(text)]


def _kinds(text: str) -> list[str]:
    """Return just the kind of each token in *text*."""
    return [t.kind for t in tokenize(text)]


def _rawblocks(toks: list[Tok]) -> list[Tok]:
    """Return every RAWBLOCK token from *toks*."""
    return [t for t in toks if t.kind == "RAWBLOCK"]


def _first(toks: list[Tok], kind: str) -> Tok:
    """Return the first token with the given *kind*."""
    return next(t for t in toks if t.kind == kind)


def _find_kw(toks: list[Tok], kw: str) -> Tok:
    """Return the first KW token whose value equals *kw*."""
    return next(t for t in toks if t.kind == "KW" and t.value == kw)


# ===========================================================================
# Keywords
# ===========================================================================

class TestKeywords:
    """Every reserved word produces a single KW token."""

    def test_each_keyword_is_kw(self) -> None:
        # 'then' is omitted: a bare 'then' is an unterminated block (SyntaxError).
        # It is tested in context in TestRawBlock.
        for word in (
            "package", "declare", "rule", "end", "when",
            "not", "salience", "true", "false", "null", "None",
        ):
            assert _kv(word) == [("KW", word)], f"failed for keyword {word!r}"

    def test_no_loop_is_single_kw_token(self) -> None:
        assert _kv("no-loop") == [("KW", "no-loop")]

    def test_not_is_not_mangled_by_no_loop_rule(self) -> None:
        assert _kv("not") == [("KW", "not")]

    def test_keyword_followed_by_identifier(self) -> None:
        assert _kv("rule MyRule") == [("KW", "rule"), ("IDENT", "MyRule")]


# ===========================================================================
# Identifiers
# ===========================================================================

class TestIdentifiers:
    """Non-keyword words produce IDENT tokens."""

    def test_capitalized_type_name(self) -> None:
        assert _kv("MyType") == [("IDENT", "MyType")]

    def test_underscore_prefix(self) -> None:
        assert _kv("_temp") == [("IDENT", "_temp")]

    def test_alphanumeric(self) -> None:
        assert _kv("name123") == [("IDENT", "name123")]

    def test_lowercase_non_keyword(self) -> None:
        assert _kv("value") == [("IDENT", "value")]


# ===========================================================================
# Variables
# ===========================================================================

class TestVariables:
    """Dollar-prefixed names produce VAR tokens."""

    def test_single_char_var(self) -> None:
        assert _kv("$x") == [("VAR", "$x")]

    def test_multi_char_var(self) -> None:
        assert _kv("$myVar") == [("VAR", "$myVar")]

    def test_var_with_underscore(self) -> None:
        assert _kv("$my_var") == [("VAR", "$my_var")]

    def test_var_in_binding_context(self) -> None:
        result = _kv("$a: name")
        assert result[0] == ("VAR", "$a")
        assert result[1] == ("PUNCT", ":")


# ===========================================================================
# Literals
# ===========================================================================

class TestLiterals:
    """String, integer, and float literals are correctly recognised."""

    def test_double_quoted_string(self) -> None:
        assert _kv('"hello"') == [("STRING", '"hello"')]

    def test_single_quoted_string(self) -> None:
        assert _kv("'world'") == [("STRING", "'world'")]

    def test_string_with_escaped_double_quote(self) -> None:
        raw = r'"say \"hi\""'
        assert _kv(raw) == [("STRING", raw)]

    def test_integer(self) -> None:
        assert _kv("42") == [("INT", "42")]

    def test_float(self) -> None:
        assert _kv("3.14") == [("FLOAT", "3.14")]

    def test_float_not_split_as_int_dot_int(self) -> None:
        assert _kinds("3.14") == ["FLOAT"]

    def test_true_false_null_none_are_keywords(self) -> None:
        for kw in ("true", "false", "null", "None"):
            assert _kinds(kw) == ["KW"], f"expected KW for {kw!r}"


# ===========================================================================
# Operators
# ===========================================================================

class TestOperators:
    """Comparison operators produce OP tokens."""

    def test_each_operator(self) -> None:
        for op in ("==", "!=", "<=", ">=", "<", ">"):
            assert _kv(op) == [("OP", op)], f"failed for operator {op!r}"

    def test_lte_not_split(self) -> None:
        assert _kv("<=") == [("OP", "<=")]

    def test_gte_not_split(self) -> None:
        assert _kv(">=") == [("OP", ">=")]

    def test_lt_followed_by_space_and_int(self) -> None:
        assert _kv("< 5") == [("OP", "<"), ("INT", "5")]


# ===========================================================================
# Punctuation
# ===========================================================================

class TestPunctuation:
    """Each single-character punctuation mark produces a PUNCT token."""

    def test_each_punct_char(self) -> None:
        for ch in ",./()[]{};:-":
            assert _kv(ch) == [("PUNCT", ch)], f"failed for {ch!r}"

    def test_hyphen_is_punct_not_part_of_int(self) -> None:
        # '-' standalone is PUNCT; the parser assembles negative literals
        assert _kv("-") == [("PUNCT", "-")]

    def test_field_path_dot(self) -> None:
        result = _kv("address.city")
        assert result == [("IDENT", "address"), ("PUNCT", "."), ("IDENT", "city")]


# ===========================================================================
# Comments
# ===========================================================================

class TestComments:
    """Comments produce no tokens."""

    def test_line_comment_produces_no_token(self) -> None:
        assert _kv("// this is a comment\n") == []

    def test_block_comment_produces_no_token(self) -> None:
        assert _kv("/* block */") == []

    def test_line_comment_strips_mid_line(self) -> None:
        result = _kv("declare // strip me\nPerson")
        assert result == [("KW", "declare"), ("IDENT", "Person")]

    def test_block_comment_strips_inline(self) -> None:
        result = _kv("rule /* skip */ end")
        assert result == [("KW", "rule"), ("KW", "end")]

    def test_multiline_block_comment(self) -> None:
        result = _kv("/* line1\nline2 */declare")
        assert result == [("KW", "declare")]


# ===========================================================================
# no-loop keyword
# ===========================================================================

class TestNoLoop:
    """``no-loop`` is a two-word keyword joined by a hyphen."""

    def test_no_loop_single_token(self) -> None:
        assert _kv("no-loop") == [("KW", "no-loop")]

    def test_no_loop_followed_by_bool(self) -> None:
        assert _kv("no-loop true") == [("KW", "no-loop"), ("KW", "true")]

    def test_no_loop_preceded_by_not(self) -> None:
        result = _kv("not no-loop")
        assert result == [("KW", "not"), ("KW", "no-loop")]

    def test_no_alone_is_ident(self) -> None:
        assert _kv("no") == [("IDENT", "no")]


# ===========================================================================
# Raw block (then-block)
# ===========================================================================

class TestRawBlock:
    """The then-block is captured verbatim as a single RAWBLOCK token."""

    def test_simple_then_block_structure(self) -> None:
        toks = tokenize("then\n  x = 1\nend")
        assert len(toks) == 3
        assert toks[0] == Tok("KW", "then", 1)
        assert toks[1].kind == "RAWBLOCK"
        assert toks[1].value == "  x = 1\n"

    def test_simple_then_block_end_token(self) -> None:
        toks = tokenize("then\n  x = 1\nend")
        assert toks[2].kind == "KW"
        assert toks[2].value == "end"

    def test_empty_then_block(self) -> None:
        toks = tokenize("then\nend")
        assert toks[1].kind == "RAWBLOCK"
        assert toks[1].value == ""

    def test_end_flag_variable_does_not_close_block(self) -> None:
        toks = tokenize("then\n  end_flag = True\nend")
        assert toks[1].kind == "RAWBLOCK"
        assert "end_flag" in toks[1].value

    def test_end_method_call_does_not_close_block(self) -> None:
        toks = tokenize("then\n  obj.end_session()\nend")
        assert toks[1].kind == "RAWBLOCK"
        assert "end_session" in toks[1].value

    def test_indented_end_closes_block(self) -> None:
        toks = tokenize("then\n  pass\n  end\n")
        assert toks[1].kind == "RAWBLOCK"
        assert toks[2].value == "end"

    def test_end_with_trailing_whitespace_closes_block(self) -> None:
        toks = tokenize("then\n  pass\nend   \n")
        assert toks[2].value == "end"

    def test_multiline_then_block(self) -> None:
        src = "then\n  a = 1\n  b = 2\n  c = a + b\nend"
        toks = tokenize(src)
        assert toks[1].kind == "RAWBLOCK"
        assert toks[1].value.count("\n") == 3

    def test_unclosed_then_block_raises_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            tokenize("then\n  pass\n")

    def test_dollar_sign_preserved_in_rawblock(self) -> None:
        toks = tokenize("then\n  $app.ok = False\nend")
        assert "$app" in toks[1].value


# ===========================================================================
# Line numbers
# ===========================================================================

class TestLineNumbers:
    """Tokens carry the correct 1-based source line number."""

    def test_first_token_is_line_one(self) -> None:
        toks = tokenize("rule")
        assert toks[0].line == 1

    def test_token_on_second_line(self) -> None:
        toks = tokenize("rule\nwhen")
        assert toks[0].line == 1
        assert toks[1].line == 2

    def test_rawblock_carries_then_line_number(self) -> None:
        toks = tokenize("rule\nwhen\nthen\n  pass\nend")
        then_tok = _find_kw(toks, "then")
        raw_tok = _first(toks, "RAWBLOCK")
        assert then_tok.line == raw_tok.line

    def test_tokens_after_rawblock_have_correct_line(self) -> None:
        # 'end' closes the block on line 5; next rule starts on line 6
        src = "then\n  x=1\n  y=2\n  z=3\nend\nrule"
        toks = tokenize(src)
        rule_tok = next(t for t in toks if t.kind == "KW" and t.value == "rule")
        assert rule_tok.line == 6


# ===========================================================================
# End-to-end
# ===========================================================================

class TestDeclareBlock:
    """Declare block tokenizes to the expected flat sequence."""

    def test_declare_block_header(self) -> None:
        result = _kv("declare Temp\n  value: float\nend")
        assert result[0] == ("KW", "declare")
        assert result[1] == ("IDENT", "Temp")
        assert result[-1] == ("KW", "end")

    def test_declare_block_field_tokens(self) -> None:
        result = _kv("declare Temp\n  value: float\nend")
        assert ("IDENT", "value") in result
        assert ("PUNCT", ":") in result


class TestRuleBlock:
    """Single rule program tokenizes correctly end-to-end."""

    def test_rule_starts_with_kw(self) -> None:
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "too-hot"\n  when\n    /Temp[value >= 80]\n  then\n    pass\nend\n'
        )
        toks = tokenize(src)
        assert toks[0].kind == "KW"

    def test_rule_contains_one_rawblock(self) -> None:
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "too-hot"\n  when\n    /Temp[value >= 80]\n  then\n    pass\nend\n'
        )
        rawblocks = _rawblocks(tokenize(src))
        assert len(rawblocks) == 1
        assert "pass" in rawblocks[0].value


class TestMultiRule:
    """Multi-rule programs yield one RAWBLOCK per rule."""

    def test_two_rules_produce_two_rawblocks(self) -> None:
        src = (
            'rule "a"\n  when\n  then\n    x = 1\nend\n'
            'rule "b"\n  when\n  then\n    y = 2\nend\n'
        )
        rawblocks = _rawblocks(tokenize(src))
        assert len(rawblocks) == 2

    def test_rawblock_content_is_per_rule(self) -> None:
        src = (
            'rule "a"\n  when\n  then\n    x = 1\nend\n'
            'rule "b"\n  when\n  then\n    y = 2\nend\n'
        )
        rawblocks = _rawblocks(tokenize(src))
        assert "x = 1" in rawblocks[0].value
        assert "y = 2" in rawblocks[1].value
