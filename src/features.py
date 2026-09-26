import polars as pl
from rapidfuzz import fuzz

def _jaro_winkler_sim(s: pl.Series) -> pl.Series:
    """Computes Jaro-Winkler similarity for a struct series of two string columns."""
    col1 = s.struct.field(s.struct.fields[0])
    col2 = s.struct.field(s.struct.fields[1])
    # Handle None properly
    return pl.Series([
        fuzz.ratio(a, b) / 100.0 if a is not None and b is not None else 0.0
        for a, b in zip(col1, col2)
    ])

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes ML features for candidate pairs.
    Expects df to contain _s1 and _cand suffixed columns for:
    - norm_name_clean
    - norm_address
    - country
    - postal_code
    """
    
    df = df.with_columns(
        name_exact_match=(pl.col("norm_name_clean_s1") == pl.col("norm_name_clean_cand")).cast(pl.Int8).fill_null(0),
        address_exact_match=(pl.col("norm_address_s1") == pl.col("norm_address_cand")).cast(pl.Int8).fill_null(0),
        
        country_agree=(pl.col("country_s1") == pl.col("country_cand")).cast(pl.Int8).fill_null(0),
        postal_agree=(pl.col("postal_code_s1") == pl.col("postal_code_cand")).cast(pl.Int8).fill_null(0),
        
        name_len_diff=(pl.col("norm_name_clean_s1").cast(pl.String).str.len_chars().cast(pl.Int32) - pl.col("norm_name_clean_cand").cast(pl.String).str.len_chars().cast(pl.Int32)).abs().fill_null(999),
        address_len_diff=(pl.col("norm_address_s1").cast(pl.String).str.len_chars().cast(pl.Int32) - pl.col("norm_address_cand").cast(pl.String).str.len_chars().cast(pl.Int32)).abs().fill_null(999),
        
        # Token overlap setup
        name_tok_s1=pl.col("norm_name_clean_s1").cast(pl.String).fill_null("").str.split(" "),
        name_tok_cand=pl.col("norm_name_clean_cand").cast(pl.String).fill_null("").str.split(" "),
        addr_tok_s1=pl.col("norm_address_s1").cast(pl.String).fill_null("").str.split(" "),
        addr_tok_cand=pl.col("norm_address_cand").cast(pl.String).fill_null("").str.split(" "),
    )
    
    df = df.with_columns(
        name_tok_overlap=pl.col("name_tok_s1").list.set_intersection(pl.col("name_tok_cand")).list.len().fill_null(0),
        addr_tok_overlap=pl.col("addr_tok_s1").list.set_intersection(pl.col("addr_tok_cand")).list.len().fill_null(0),
    )
    
    # Calculate Jaccard and drop intermediate token columns
    df = df.with_columns(
        name_tok_jaccard=(
            pl.col("name_tok_overlap") / 
            (pl.col("name_tok_s1").list.len() + pl.col("name_tok_cand").list.len() - pl.col("name_tok_overlap"))
        ).fill_null(0.0),
        addr_tok_jaccard=(
            pl.col("addr_tok_overlap") / 
            (pl.col("addr_tok_s1").list.len() + pl.col("addr_tok_cand").list.len() - pl.col("addr_tok_overlap"))
        ).fill_null(0.0),
        
        name_jaro_winkler=pl.struct(["norm_name_clean_s1", "norm_name_clean_cand"]).map_batches(
            _jaro_winkler_sim, return_dtype=pl.Float64
        ),
        addr_jaro_winkler=pl.struct(["norm_address_s1", "norm_address_cand"]).map_batches(
            _jaro_winkler_sim, return_dtype=pl.Float64
        ),
    )
    
    # Clean up intermediate columns
    df = df.drop(["name_tok_s1", "name_tok_cand", "addr_tok_s1", "addr_tok_cand"])
    
    return df
