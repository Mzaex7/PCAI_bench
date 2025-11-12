#!/usr/bin/env python3
"""
Generate visualizations from existing benchmark CSV results.

This script allows you to regenerate charts from previously saved benchmark data
without re-running the entire benchmark. Supports combining multiple CSVs for
comparative analysis.
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from visualize import BenchmarkVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations from benchmark CSV results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate visualizations from the latest results
  python generate_visualizations.py
  
  # Generate from a specific CSV file
  python generate_visualizations.py --csv results/llama.csv
  
  # Combine multiple CSV files (e.g., different concurrency levels)
  python generate_visualizations.py --csv results/llama.csv results/llama-5.csv results/llama-10.csv
  
  # Use a pattern to match multiple files
  python generate_visualizations.py --pattern "results/llama*.csv"
  
  # Compare different models across concurrency levels
  python generate_visualizations.py --pattern "results/llama-*.csv" --output-dir viz/llama_comparison
  
  # Combine Ollama benchmarks
  python generate_visualizations.py --pattern "results/*-ollama*.csv" --output-dir viz/ollama_analysis
        """
    )
    
    parser.add_argument(
        '--csv',
        nargs='+',
        type=str,
        help='Path(s) to CSV file(s) with benchmark results. Can specify multiple files to combine them.'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        help='Glob pattern to match multiple CSV files (e.g., "results/llama*.csv")'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for visualizations (default: same directory as CSV file or "results/viz")'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        help='Custom title prefix for visualizations (default: auto-detected from files)'
    )
    
    return parser.parse_args()


def find_csv_files(pattern: str) -> List[str]:
    """Find CSV files matching a glob pattern."""
    from glob import glob
    
    files = glob(pattern)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    return sorted(csv_files)


def find_latest_csv(results_dir='results'):
    """Find the most recent benchmark CSV file."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return None
    
    csv_files = list(results_path.glob('benchmark_results_*.csv'))
    
    if not csv_files:
        return None
    
    # Sort by modification time, most recent first
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    return str(latest_csv)


def load_and_combine_csvs(csv_files: List[str]) -> pd.DataFrame:
    """
    Load and combine multiple CSV files into a single DataFrame.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        Combined DataFrame
    """
    print(f"\n📂 Loading CSV files...")
    
    dataframes = []
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"  ⚠️  Warning: File not found: {csv_file}")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
            print(f"  ✓ Loaded: {csv_file} ({len(df)} records)")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file}: {e}")
    
    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")
    
    # Combine all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\n  📊 Total records: {len(combined_df)}")
    print(f"  📊 Unique endpoints: {combined_df['endpoint_name'].nunique()}")
    print(f"  📊 Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    return combined_df


def detect_comparison_type(csv_files: List[str]) -> str:
    """Detect what type of comparison is being made based on filenames."""
    basenames = [os.path.basename(f) for f in csv_files]
    
    # Check for concurrency comparison (llama.csv, llama-5.csv, llama-10.csv, etc.)
    if any('-' in name and name.split('-')[-1].replace('.csv', '').isdigit() for name in basenames):
        return "concurrency"
    
    # Check for engine comparison (llama-vllm, llama-nim, etc.)
    if any('vllm' in name.lower() or 'nim' in name.lower() or 'ollama' in name.lower() for name in basenames):
        return "engine"
    
    # Check for model comparison (llama, qwen, gemma, etc.)
    unique_prefixes = set()
    for name in basenames:
        prefix = name.split('-')[0].split('.')[0]
        unique_prefixes.add(prefix)
    
    if len(unique_prefixes) > 1:
        return "model"
    
    return "general"


def main():
    """Main execution function."""
    args = parse_args()
    
    # Determine which CSV files to process
    csv_files = []
    
    if args.pattern:
        print(f"🔍 Searching for files matching pattern: {args.pattern}")
        csv_files = find_csv_files(args.pattern)
        
        if not csv_files:
            print(f"❌ Error: No files found matching pattern: {args.pattern}")
            sys.exit(1)
        
        print(f"✓ Found {len(csv_files)} file(s):")
        for f in csv_files:
            print(f"    • {f}")
    
    elif args.csv:
        csv_files = args.csv
        
        # Validate files exist
        for f in csv_files:
            if not os.path.exists(f):
                print(f"❌ Error: CSV file not found: {f}")
                sys.exit(1)
    
    else:
        print("No CSV file specified, looking for most recent benchmark results...")
        latest = find_latest_csv()
        
        if not latest:
            print("❌ Error: No benchmark results found in 'results/' directory")
            print("   Please specify a CSV file with --csv or use --pattern")
            sys.exit(1)
        
        csv_files = [latest]
        print(f"✓ Found: {latest}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use same directory as first CSV file, or results/viz
        if csv_files:
            output_dir = os.path.join(os.path.dirname(csv_files[0]), 'viz')
        else:
            output_dir = 'results/viz'
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and combine CSV files
    try:
        combined_df = load_and_combine_csvs(csv_files)
    except Exception as e:
        print(f"\n❌ Error loading CSV files: {e}")
        sys.exit(1)
    
    # Detect comparison type
    comparison_type = detect_comparison_type(csv_files)
    
    print(f"\n📊 Generating visualizations...")
    print(f"   Comparison type:  {comparison_type}")
    print(f"   CSV files:        {len(csv_files)}")
    print(f"   Output dir:       {output_dir}")
    
    # Custom title if provided
    if args.title:
        print(f"   Title prefix:     {args.title}")
    
    print()
    
    # Convert DataFrame to list of dicts for BenchmarkVisualizer
    results = combined_df.to_dict('records')
    
    # Generate visualizations
    try:
        visualizer = BenchmarkVisualizer(results, output_dir)
        
        # Update figure title if custom title provided
        if args.title:
            # This would require modifying BenchmarkVisualizer to accept title parameter
            # For now, just use default titles
            pass
        
        visualizer.generate_all_visualizations()
        
        print("\n✅ Visualizations generated successfully!")
        print(f"   Charts saved to: {output_dir}/")
        print(f"\n💡 Tips:")
        print(f"   • Check {output_dir}/01_engine_comparison_*.png for overview")
        print(f"   • All {len(csv_files)} CSV file(s) have been combined for analysis")
        
        if comparison_type == "concurrency":
            print(f"   • Charts show performance across different concurrency levels")
        elif comparison_type == "engine":
            print(f"   • Charts compare different inference engines (vLLM, NIM, Ollama)")
        elif comparison_type == "model":
            print(f"   • Charts compare different model families")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
