"""
Phase 1 Blocking – Simple deterministic multi-pass candidate generator.

Strategy (no IDF, no inverted-index, no out-of-core complexity):
  Pass A: country + exact norm_name_clean
  Pass B: country + name-prefix4 (first 4 chars of name, spaces stripped)
  Pass C: country + name token (each significant word in norm_name_clean)
  Pass D: country + postal_code + name token  (where postal_code available)
  Pass E: country + address digit + name token (house/street number overlap)

All passes use vectorised Polars joins.  Memory stays bounded because we
process the candidate files in row-group chunks and never build a global
inverted index.

Public API
----------
build_keys(df) -> pl.DataFrame
    columns: entity_id, block_key, pass_name

generate_candidates(s1_df, cand_df) -> pl.DataFrame
    columns: source1_entity_id, candidate_entity_id, passes (list[str]),
             n_passes (int)
"""

from __future__ import annotations
import polars as pl

# Minimum token length for name/address word blocking
_MIN_NAME_TOK_LEN = 3

# Stop tokens that add no discrimination power even after normalization
_NAME_STOP = frozenset({
    "the", "and", "for", "its", "inc", "llc", "ltd", "pvt", "plc",
    "corp", "co", "sa", "sas", "sarl", "gmbh", "bv", "nv", "ab",
    "pty", "srl", "spa", "oe", "ou", "oy", "as", "ag", "kg", "kk",
    "lp", "llp", "of", "in", "at", "by", "to", "is", "or", "an",
    "de", "du", "la", "le", "les", "et", "en", "sur",
    "private", "public", "limited", "company",
})


# ── Utility helpers ────────────────────────────────────────────────────────────

def _name_tokens_expr(col: str = "norm_name_clean") -> pl.Expr:
    """Return explodable list of significant name tokens from *col*."""
    return (
        pl.col(col)
        .str.split(" ")
        .list.eval(
            pl.element()
            .filter(
                (pl.element().str.len_bytes() >= _MIN_NAME_TOK_LEN) &
                (~pl.element().is_in(list(_NAME_STOP)))
            )
        )
    )


def _prefix4_expr(col: str = "norm_name_clean") -> pl.Expr:
    """4-char prefix of name with spaces stripped."""
    return (
        pl.col(col)
        .str.replace_all(r"[\s_]+", "")
        .str.slice(0, 4)
    )


def _addr_digits_expr(col: str = "norm_address") -> pl.Expr:
    """All digit tokens (>=2 chars) extracted from address."""
    return (
        pl.col(col)
        .str.extract_all(r"\b\d{2,}\b")
    )


def _empty_keys() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"entity_id": pl.String, "block_key": pl.String, "pass_name": pl.String}
    )


# ── Per-pass key builders ──────────────────────────────────────────────────────

def _build_exact_name_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Pass A: country|exact_norm_name_clean."""
    required = {"entity_id", "country", "norm_name_clean"}
    if not required.issubset(df.columns):
        return _empty_keys()
    return (
        df.select(["entity_id", "country", "norm_name_clean"])
        .filter(
            pl.col("norm_name_clean").is_not_null() &
            (pl.col("norm_name_clean").str.len_bytes() >= 3)
        )
        .with_columns(
            pl.concat_str(
                [pl.col("country"), pl.lit("|A|"), pl.col("norm_name_clean")]
            ).alias("block_key"),
            pl.lit("A").alias("pass_name"),
        )
        .select(["entity_id", "block_key", "pass_name"])
        .unique(["entity_id", "block_key"])
    )


def _build_prefix_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Pass B: country|prefix4."""
    required = {"entity_id", "country", "norm_name_clean"}
    if not required.issubset(df.columns):
        return _empty_keys()
    return (
        df.select(["entity_id", "country", "norm_name_clean"])
        .filter(
            pl.col("norm_name_clean").is_not_null() &
            (pl.col("norm_name_clean").str.len_bytes() >= 3)
        )
        .with_columns(_prefix4_expr().alias("pfx"))
        .filter(pl.col("pfx").str.len_bytes() >= 3)
        .with_columns(
            pl.concat_str(
                [pl.col("country"), pl.lit("|B|"), pl.col("pfx")]
            ).alias("block_key"),
            pl.lit("B").alias("pass_name"),
        )
        .select(["entity_id", "block_key", "pass_name"])
        .unique(["entity_id", "block_key"])
    )


def _build_name_token_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Pass C: country|name_token (one row per token)."""
    required = {"entity_id", "country", "norm_name_clean"}
    if not required.issubset(df.columns):
        return _empty_keys()
    return (
        df.select(["entity_id", "country", "norm_name_clean"])
        .filter(pl.col("norm_name_clean").is_not_null())
        .with_columns(_name_tokens_expr().alias("tok"))
        .explode("tok")
        .drop_nulls("tok")
        .filter(pl.col("tok").str.len_bytes() >= _MIN_NAME_TOK_LEN)
        .with_columns(
            pl.concat_str(
                [pl.col("country"), pl.lit("|C|"), pl.col("tok")]
            ).alias("block_key"),
            pl.lit("C").alias("pass_name"),
        )
        .select(["entity_id", "block_key", "pass_name"])
        .unique(["entity_id", "block_key"])
    )


def _build_postal_name_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Pass D: country|postal_code|name_token (where postal_code exists)."""
    required = {"entity_id", "country", "postal_code", "norm_name_clean"}
    if not required.issubset(df.columns):
        return _empty_keys()
    return (
        df.select(["entity_id", "country", "postal_code", "norm_name_clean"])
        .filter(
            pl.col("postal_code").is_not_null() &
            (pl.col("postal_code").str.len_bytes() >= 4) &
            pl.col("norm_name_clean").is_not_null()
        )
        .with_columns(_name_tokens_expr().alias("tok"))
        .explode("tok")
        .drop_nulls("tok")
        .filter(pl.col("tok").str.len_bytes() >= _MIN_NAME_TOK_LEN)
        .with_columns(
            pl.concat_str(
                [pl.col("country"), pl.lit("|D|"),
                 pl.col("postal_code"), pl.lit("|"), pl.col("tok")]
            ).alias("block_key"),
            pl.lit("D").alias("pass_name"),
        )
        .select(["entity_id", "block_key", "pass_name"])
        .unique(["entity_id", "block_key"])
    )


def _build_addrnum_name_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Pass E: country|addr_digit|name_token (where address digit available)."""
    required = {"entity_id", "country", "norm_address", "norm_name_clean"}
    if not required.issubset(df.columns):
        return _empty_keys()
    return (
        df.select(["entity_id", "country", "norm_address", "norm_name_clean"])
        .filter(
            pl.col("norm_address").is_not_null() &
            pl.col("norm_name_clean").is_not_null()
        )
        .with_columns(_addr_digits_expr().alias("dig"))
        .explode("dig")
        .drop_nulls("dig")
        .with_columns(_name_tokens_expr().alias("tok"))
        .explode("tok")
        .drop_nulls("tok")
        .filter(
            pl.col("tok").str.len_bytes() >= _MIN_NAME_TOK_LEN
        )
        .with_columns(
            pl.concat_str(
                [pl.col("country"), pl.lit("|E|"),
                 pl.col("dig"), pl.lit("|"), pl.col("tok")]
            ).alias("block_key"),
            pl.lit("E").alias("pass_name"),
        )
        .select(["entity_id", "block_key", "pass_name"])
        .unique(["entity_id", "block_key"])
    )


# ── Public API ─────────────────────────────────────────────────────────────────

PASS_BUILDERS = [
    ("A", _build_exact_name_keys),
    ("B", _build_prefix_keys),
    ("C", _build_name_token_keys),
    ("D", _build_postal_name_keys),
    ("E", _build_addrnum_name_keys),
]


def build_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Build all blocking keys for *df* (any source).

    Returns a frame with columns: entity_id, block_key, pass_name.
    """
    chunks = [fn(df) for _, fn in PASS_BUILDERS]
    non_empty = [c for c in chunks if len(c) > 0]
    if not non_empty:
        return _empty_keys()
    return pl.concat(non_empty).unique(["entity_id", "block_key"])


def generate_candidates(
    s1_df: pl.DataFrame,
    cand_df: pl.DataFrame,
    max_key_freq: int = 50
) -> pl.DataFrame:
    """Join S1 keys against candidate keys and return candidate pairs.

    Parameters
    ----------
    s1_df:
        DataFrame from train_source1 normalized parquet (full schema).
    cand_df:
        DataFrame from train_source2 or train_source3 normalized parquet.
    max_key_freq:
        Maximum occurrences of a block key in cand_df. If a key appears
        more than this, it is discarded (acts as a dynamic stop-word filter).

    Returns
    -------
    pl.DataFrame with columns:
        source1_entity_id, candidate_entity_id, passes (list[str]), n_passes (int)
    """
    s1_keys = build_keys(s1_df).rename({"entity_id": "s1_id"})
    cand_keys = build_keys(cand_df).rename(
        {"entity_id": "cand_id", "pass_name": "cand_pass"}
    )

    if max_key_freq > 0:
        # Prevent cross-join explosion from highly common tokens
        cand_keys = cand_keys.filter(pl.len().over("block_key") <= max_key_freq)

    joined = (
        s1_keys.join(cand_keys, on="block_key", how="inner")
        .select(["s1_id", "cand_id", "pass_name"])
        .unique(["s1_id", "cand_id", "pass_name"])
    )

    if len(joined) == 0:
        return pl.DataFrame(
            schema={
                "source1_entity_id": pl.String,
                "candidate_entity_id": pl.String,
                "passes": pl.List(pl.String),
                "n_passes": pl.Int32,
            }
        )

    result = (
        joined.group_by(["s1_id", "cand_id"])
        .agg(
            pl.col("pass_name").unique().sort().alias("passes"),
            pl.col("pass_name").n_unique().cast(pl.Int32).alias("n_passes"),
        )
        .rename({"s1_id": "source1_entity_id", "cand_id": "candidate_entity_id"})
    )
    return result
