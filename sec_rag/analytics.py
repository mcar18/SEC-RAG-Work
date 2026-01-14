"""Analytics and visualization for risk themes."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sec_rag.config import OUTPUTS_DIR
from sec_rag.themes import score_themes_multiple


def plot_theme_timeline(
    df: pd.DataFrame,
    theme: str,
    tickers: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot theme score over time.
    
    Args:
        df: DataFrame from score_themes_multiple
        theme: Theme name to plot
        tickers: Optional list of tickers to filter
        output_path: Optional output path
    """
    theme_df = df[df["theme"] == theme].copy()
    
    if tickers:
        theme_df = theme_df[theme_df["ticker"].isin(tickers)]
    
    if theme_df.empty:
        print(f"No data for theme {theme}")
        return
    
    plt.figure(figsize=(12, 6))
    
    for ticker in theme_df["ticker"].unique():
        ticker_data = theme_df[theme_df["ticker"] == ticker].sort_values("year")
        # Convert scores to percentages
        scores_pct = ticker_data["score"] * 100
        plt.plot(ticker_data["year"], scores_pct, marker="o", label=ticker, linewidth=2)
    
    # Format x-axis as integers (no decimals)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1f}%'))
    
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Theme Score (%)", fontsize=12)
    plt.title(f"{theme} Risk Theme Over Time", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.savefig(OUTPUTS_DIR / f"theme_timeline_{theme.replace('/', '_')}.png", dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_theme_heatmap(
    df: pd.DataFrame,
    year: int,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot heatmap of themes vs companies for a given year.
    
    Args:
        df: DataFrame from score_themes_multiple
        year: Year to plot
        output_path: Optional output path
    """
    year_df = df[df["year"] == year].copy()
    
    if year_df.empty:
        print(f"No data for year {year}")
        return
    
    pivot = year_df.pivot_table(index="ticker", columns="theme", values="score", aggfunc="mean")
    
    plt.figure(figsize=(14, max(6, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": "Theme Score"})
    plt.title(f"Risk Theme Scores by Company ({year})", fontsize=14, fontweight="bold")
    plt.xlabel("Theme", fontsize=12)
    plt.ylabel("Ticker", fontsize=12)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.savefig(OUTPUTS_DIR / f"theme_heatmap_{year}.png", dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_top_movers(
    df: pd.DataFrame,
    n: int = 10,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot top movers (largest year-over-year changes).
    
    Args:
        df: DataFrame from score_themes_multiple
        n: Number of top movers to show
        output_path: Optional output path
    """
    # Calculate year-over-year changes
    df_sorted = df.sort_values(["ticker", "theme", "year"])
    df_sorted["prev_score"] = df_sorted.groupby(["ticker", "theme"])["score"].shift(1)
    df_sorted["prev_year"] = df_sorted.groupby(["ticker", "theme"])["year"].shift(1)
    df_sorted["yoy_change"] = df_sorted["score"] - df_sorted["prev_score"]
    df_sorted["has_prev"] = df_sorted["prev_score"].notna()
    
    # Get top movers (absolute change)
    movers = df_sorted[df_sorted["has_prev"]].copy()
    movers["abs_change"] = movers["yoy_change"].abs()
    top_movers = movers.nlargest(n, "abs_change")
    
    if top_movers.empty:
        print("No year-over-year data available")
        return
    
    plt.figure(figsize=(12, max(6, n * 0.5)))
    colors = ["red" if x < 0 else "green" for x in top_movers["yoy_change"]]
    plt.barh(
        range(len(top_movers)),
        top_movers["yoy_change"],
        color=colors,
        alpha=0.7
    )
    plt.yticks(range(len(top_movers)), [
        f"{row['ticker']} - {row['theme']} ({row['prev_year']}→{row['year']})"
        for _, row in top_movers.iterrows()
    ])
    plt.xlabel("Year-over-Year Change", fontsize=12)
    plt.title(f"Top {n} Theme Score Movers", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.savefig(OUTPUTS_DIR / "top_movers.png", dpi=300, bbox_inches="tight")
    
    plt.close()


def generate_analytics(
    filings_data: dict,
    tickers: List[str],
    years: List[int],
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate all analytics and save to disk.
    
    Args:
        filings_data: Nested dict {ticker: {year: (metadata, chunks, index)}}
        tickers: List of tickers
        years: List of years
        output_dir: Optional output directory
        
    Returns:
        DataFrame with theme scores
    """
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Score themes
    print("Scoring risk themes...")
    df = score_themes_multiple(filings_data)
    
    # Save CSV
    csv_path = output_dir / "theme_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved theme scores to {csv_path}")
    
    # Generate plots
    print("Generating visualizations...")
    
    # Timeline plots for each theme
    for theme in df["theme"].unique():
        plot_theme_timeline(df, theme, tickers=tickers, output_path=output_dir / f"timeline_{theme.replace('/', '_')}.png")
    
    # Heatmap for latest year
    if years:
        latest_year = max(years)
        plot_theme_heatmap(df, latest_year, output_path=output_dir / f"heatmap_{latest_year}.png")
    
    # Top movers
    plot_top_movers(df, output_path=output_dir / "top_movers.png")
    
    return df

