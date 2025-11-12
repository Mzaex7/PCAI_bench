#!/usr/bin/env python3
"""
Special analysis for Ollama's concurrent request behavior.
Investigates why Ollama performs poorly under high concurrency.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os
import sys

# HPE Corporate Color Palette
COLORS = {
    'ollama': '#01A982',       # HPE Green
    'vllm': '#7764FC',         # HPE Purple
    'nim': '#0070F8',          # HPE Blue
    'danger': '#FF4444',       # Red for issues
    'warning': '#FFA500',      # Orange
    'success': '#01A982',      # Green
}

def load_ollama_data():
    """Load all Ollama concurrent data files."""
    files = [
        'results/llama-ollama.csv',
        'results/llama-ollama-5.csv',
        'results/llama-ollama-10.csv',
        'results/llama-ollama-25.csv',
        'results/llama-ollama-50.csv',
    ]
    
    dfs = []
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
            print(f"✓ Loaded {file}: {len(df)} records")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Total records: {len(combined)}")
    print(f"📊 Concurrency levels: {sorted(combined['concurrent_level'].unique())}")
    
    return combined

def analyze_request_timing(df):
    """Analyze how requests are actually being processed over time."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Ollama Concurrent Request Timeline Analysis\nUnderstanding Request Processing Behavior', 
                 fontsize=16, fontweight='bold')
    
    concurrency_levels = sorted(df['concurrent_level'].unique())
    
    # 1. Request completion timeline for each concurrency level
    ax1 = axes[0, 0]
    for c_level in concurrency_levels:
        df_c = df[df['concurrent_level'] == c_level].copy()
        df_c = df_c.sort_values('timestamp')
        
        # Calculate seconds from start
        start_time = df_c['timestamp'].min()
        df_c['seconds_from_start'] = (df_c['timestamp'] - start_time).dt.total_seconds()
        
        # Plot cumulative requests
        ax1.plot(df_c['seconds_from_start'], range(1, len(df_c) + 1), 
                marker='o', markersize=3, label=f'C{c_level}', linewidth=2)
    
    ax1.set_title('Request Completion Timeline\n(Steeper = Faster Processing)', fontweight='bold', pad=10)
    ax1.set_xlabel('Time from Start (seconds)', fontweight='bold')
    ax1.set_ylabel('Cumulative Requests Completed', fontweight='bold')
    ax1.legend(title='Concurrency', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # 2. Latency distribution per concurrency level
    ax2 = axes[0, 1]
    
    data_for_box = []
    labels_for_box = []
    for c_level in concurrency_levels:
        df_c = df[df['concurrent_level'] == c_level]
        data_for_box.append(df_c['total_latency'].values)
        labels_for_box.append(f'C{c_level}')
    
    bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['ollama'])
        patch.set_alpha(0.6)
    
    ax2.set_title('Total Latency Distribution by Concurrency\n(Queuing Effects Visible)', fontweight='bold', pad=10)
    ax2.set_xlabel('Concurrency Level', fontweight='bold')
    ax2.set_ylabel('Total Latency (seconds)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add median values
    for i, c_level in enumerate(concurrency_levels, 1):
        df_c = df[df['concurrent_level'] == c_level]
        median = df_c['total_latency'].median()
        ax2.text(i, median, f'{median:.1f}s', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # 3. TTFT (Time to First Token) - Shows if server is queuing requests
    ax3 = axes[1, 0]
    
    ttft_data = []
    for c_level in concurrency_levels:
        df_c = df[df['concurrent_level'] == c_level]
        ttft_data.append(df_c['time_to_first_token'].values)
    
    bp = ax3.boxplot(ttft_data, labels=labels_for_box, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['warning'])
        patch.set_alpha(0.6)
    
    ax3.set_title('Time to First Token (TTFT) Distribution\n(High TTFT = Request Queuing)', 
                  fontweight='bold', pad=10)
    ax3.set_xlabel('Concurrency Level', fontweight='bold')
    ax3.set_ylabel('TTFT (seconds)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add median values and highlight problematic ones
    for i, c_level in enumerate(concurrency_levels, 1):
        df_c = df[df['concurrent_level'] == c_level]
        median = df_c['time_to_first_token'].median()
        color = COLORS['danger'] if median > 100 else 'black'
        ax3.text(i, median, f'{median:.1f}s', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color=color)
    
    # 4. Throughput degradation
    ax4 = axes[1, 1]
    
    throughput_stats = []
    for c_level in concurrency_levels:
        df_c = df[df['concurrent_level'] == c_level]
        mean_tps = df_c['tokens_per_second'].mean()
        std_tps = df_c['tokens_per_second'].std()
        throughput_stats.append({
            'concurrency': c_level,
            'mean': mean_tps,
            'std': std_tps
        })
    
    stats_df = pd.DataFrame(throughput_stats)
    
    bars = ax4.bar(range(len(stats_df)), stats_df['mean'], 
                   yerr=stats_df['std'], capsize=5,
                   color=COLORS['ollama'], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax4.set_xticks(range(len(stats_df)))
    ax4.set_xticklabels([f"C{c}" for c in stats_df['concurrency']])
    ax4.set_title('Average Throughput vs Concurrency\n(Should Increase, Not Decrease!)', 
                  fontweight='bold', pad=10)
    ax4.set_xlabel('Concurrency Level', fontweight='bold')
    ax4.set_ylabel('Tokens per Second (mean ± std)', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, throughput_stats)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row['std'] + 0.5,
                f"{row['mean']:.2f}", ha='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f'viz/ollama_debug_timing_analysis_{timestamp}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {filepath}")
    plt.close()

def analyze_request_overlap(df):
    """Analyze actual concurrent execution vs sequential queuing."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Ollama Concurrent Execution Analysis\nAre Requests Actually Running in Parallel?', 
                 fontsize=16, fontweight='bold')
    
    concurrency_levels = [10, 25, 50]  # Focus on high concurrency
    
    for idx, c_level in enumerate(concurrency_levels):
        if idx >= 3:
            break
            
        df_c = df[df['concurrent_level'] == c_level].copy()
        df_c = df_c.sort_values('timestamp')
        
        # Calculate when each request started and ended
        df_c['end_time'] = df_c['timestamp']
        df_c['start_time'] = df_c['end_time'] - pd.to_timedelta(df_c['total_latency'], unit='s')
        
        # Calculate overlaps
        overlaps = []
        for i in range(len(df_c)):
            row = df_c.iloc[i]
            # Count how many other requests were running at the same time
            concurrent_count = 0
            for j in range(len(df_c)):
                if i != j:
                    other = df_c.iloc[j]
                    # Check if timeframes overlap
                    if (row['start_time'] < other['end_time'] and 
                        row['end_time'] > other['start_time']):
                        concurrent_count += 1
            overlaps.append(concurrent_count)
        
        df_c['actual_concurrent'] = overlaps
        
        # Plot in one of the subplots
        ax = axes.flat[idx]
        
        # Histogram of actual concurrency
        ax.hist(df_c['actual_concurrent'], bins=range(0, max(overlaps) + 2), 
                color=COLORS['ollama'], alpha=0.7, edgecolor='black')
        ax.axvline(c_level, color=COLORS['danger'], linestyle='--', linewidth=2,
                  label=f'Expected: {c_level}')
        ax.axvline(df_c['actual_concurrent'].mean(), color=COLORS['warning'], 
                  linestyle='--', linewidth=2,
                  label=f'Actual Mean: {df_c["actual_concurrent"].mean():.1f}')
        
        ax.set_title(f'Concurrency Level {c_level}\nActual vs Expected Parallelism', 
                    fontweight='bold', pad=10)
        ax.set_xlabel('Number of Concurrent Requests', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f"Max: {df_c['actual_concurrent'].max()}\n"
        stats_text += f"Mean: {df_c['actual_concurrent'].mean():.1f}\n"
        stats_text += f"Median: {df_c['actual_concurrent'].median():.1f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9, fontweight='bold')
    
    # Last subplot: Summary comparison
    ax = axes.flat[3]
    
    summary_data = []
    for c_level in [1, 5, 10, 25, 50]:
        df_c = df[df['concurrent_level'] == c_level].copy()
        if len(df_c) > 0:
            df_c = df_c.sort_values('timestamp')
            df_c['end_time'] = df_c['timestamp']
            df_c['start_time'] = df_c['end_time'] - pd.to_timedelta(df_c['total_latency'], unit='s')
            
            overlaps = []
            for i in range(len(df_c)):
                row = df_c.iloc[i]
                concurrent_count = 0
                for j in range(len(df_c)):
                    if i != j:
                        other = df_c.iloc[j]
                        if (row['start_time'] < other['end_time'] and 
                            row['end_time'] > other['start_time']):
                            concurrent_count += 1
                overlaps.append(concurrent_count)
            
            df_c['actual_concurrent'] = overlaps
            
            summary_data.append({
                'expected': c_level,
                'actual_mean': df_c['actual_concurrent'].mean(),
                'actual_max': df_c['actual_concurrent'].max()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    x = range(len(summary_df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], summary_df['expected'], width,
                   label='Expected Concurrency', color=COLORS['danger'], alpha=0.7)
    bars2 = ax.bar([i + width/2 for i in x], summary_df['actual_mean'], width,
                   label='Actual Mean Concurrency', color=COLORS['ollama'], alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in summary_df['expected']])
    ax.set_title('Expected vs Actual Concurrent Execution\n(Gap Shows Queuing)', 
                fontweight='bold', pad=10)
    ax.set_xlabel('Configured Concurrency', fontweight='bold')
    ax.set_ylabel('Number of Concurrent Requests', fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    # Add difference annotations
    for i, row in summary_df.iterrows():
        diff = row['expected'] - row['actual_mean']
        if diff > 0:
            ax.text(i, max(row['expected'], row['actual_mean']) + 1,
                   f'-{diff:.1f}', ha='center', fontweight='bold', 
                   color=COLORS['danger'], fontsize=9)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f'viz/ollama_debug_parallelism_analysis_{timestamp}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filepath}")
    plt.close()

def main():
    print("🔍 Ollama Concurrency Debug Analysis")
    print("=" * 50)
    print("")
    
    # Create viz directory if it doesn't exist
    os.makedirs('viz', exist_ok=True)
    
    # Load data
    df = load_ollama_data()
    
    print("\n📊 Generating visualizations...")
    print("-" * 50)
    
    # Analysis 1: Request timing and throughput
    print("\n1️⃣  Analyzing request timing patterns...")
    analyze_request_timing(df)
    
    # Analysis 2: Actual concurrent execution
    print("\n2️⃣  Analyzing actual concurrent execution...")
    analyze_request_overlap(df)
    
    print("\n" + "=" * 50)
    print("✅ Analysis complete!")
    print("\n💡 Check the viz/ folder for:")
    print("   • ollama_debug_timing_analysis_*.png")
    print("   • ollama_debug_parallelism_analysis_*.png")
    print("\nThese charts will show:")
    print("   ✓ If Ollama is actually running requests in parallel")
    print("   ✓ How much queuing is happening at high concurrency")
    print("   ✓ Why performance degrades under load")

if __name__ == "__main__":
    main()
