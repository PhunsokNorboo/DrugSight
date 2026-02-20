"""DrugSight CLI entry point."""
from __future__ import annotations
import argparse
import logging
import sys

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DrugSight: AI-Powered Drug Repurposing Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python -m drugsight EFO_0000337 --demo",
    )
    parser.add_argument("disease_id", nargs="?", default="EFO_0000337",
                       help="EFO disease ID (default: Huntington's)")
    parser.add_argument("--drugbank-csv", default=None,
                       help="Path to DrugBank CSV (default: data/sample_drugbank.csv)")
    parser.add_argument("--training-csv", default=None,
                       help="Path to training CSV")
    parser.add_argument("--min-score", type=float, default=0.5,
                       help="Min association score (0-1)")
    parser.add_argument("--min-plddt", type=float, default=70.0,
                       help="Min AlphaFold confidence score")
    parser.add_argument("--top-n", type=int, default=20,
                       help="Number of top candidates to return")
    parser.add_argument("--output", default=None,
                       help="Output CSV path")
    parser.add_argument("--demo", action="store_true",
                       help="Run with sample data (no external deps)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Run pipeline
    if args.demo:
        from drugsight.pipeline import run_pipeline_demo
        result = run_pipeline_demo(disease_id=args.disease_id)
    else:
        from drugsight.pipeline import run_pipeline
        from drugsight.config import DATA_DIR
        drugbank = args.drugbank_csv or str(DATA_DIR / "sample_drugbank.csv")
        result = run_pipeline(
            disease_id=args.disease_id,
            drugbank_csv=drugbank,
            training_csv=args.training_csv,
            min_association_score=args.min_score,
            min_plddt=args.min_plddt,
            top_n=args.top_n,
        )

    # Output
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    else:
        print(f"\nTop {len(result)} Drug Repurposing Candidates:")
        print("=" * 80)
        for _, row in result.head(args.top_n).iterrows():
            rank = int(row.get("rank", 0))
            name = row.get("drug_name", "Unknown")
            score = row.get("composite_score", 0)
            factor = row.get("top_contributing_factor", "N/A")
            print(f"  #{rank:2d}  {name:<20s}  Score: {score:6.2f}  Key: {factor}")
        print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
