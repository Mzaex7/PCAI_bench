#!/usr/bin/env python3
"""
Generate visualizations from existing benchmark CSV results.

This script allows you to regenerate charts from previously saved benchmark data
without re-running the entire benchmark.
"""
import argparse
import os
import sys
from pathlib import Path

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
  python generate_visualizations.py --csv results/benchmark_results_2024-11-07_143022.csv
  
  # Generate to a custom output directory
  python generate_visualizations.py --csv results/my_results.csv --output-dir custom_charts
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to the CSV file with benchmark results. If not provided, will use the most recent CSV in results/'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for visualizations (default: same directory as CSV file)'
    )
    
    return parser.parse_args()


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


def main():
    """Main execution function."""
    args = parse_args()
    
    # Determine CSV file to use
    csv_file = args.csv
    
    if not csv_file:
        print("No CSV file specified, looking for most recent benchmark results...")
        csv_file = find_latest_csv()
        
        if not csv_file:
            print("❌ Error: No benchmark results found in 'results/' directory")
            print("   Please specify a CSV file with --csv or run a benchmark first")
            sys.exit(1)
        
        print(f"✓ Found: {csv_file}")
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found: {csv_file}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use same directory as CSV file
        output_dir = os.path.dirname(csv_file)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Generating visualizations...")
    print(f"   CSV file:      {csv_file}")
    print(f"   Output dir:    {output_dir}")
    print()
    
    # Generate visualizations
    try:
        visualizer = BenchmarkVisualizer(csv_file, output_dir)
        visualizer.generate_all_charts()
        
        print("\n✅ Visualizations generated successfully!")
        print(f"   Charts saved to: {output_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
