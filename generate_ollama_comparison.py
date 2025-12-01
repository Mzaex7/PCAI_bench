#!/usr/bin/env python3
"""
Ollama Old vs New Comprehensive Comparison Visualization.

Creates detailed visualizations comparing Ollama performance before and after
updates across all concurrency levels and model families.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Style setup
sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['figure.facecolor'] = '#FFFFFF'

# Colors
COLORS = {
    'old': '#FF6B6B',      # Red for old
    'new': '#01A982',      # HPE Green for new
    'llama': '#0070F8',    # Blue
    'qwen': '#7764FC',     # Purple
    'gemma': '#FF9500',    # Orange
}

def load_all_ollama_data():
    """Load all Ollama CSV files and structure them."""
    models = ['llama', 'qwen', 'gemma']
    concurrency_levels = ['sequential', '5', '10', '25', '50']
    
    data = []
    
    for model in models:
        for conc in concurrency_levels:
            if conc == 'sequential':
                old_file = f'results/{model}-ollama.csv'
                new_file = f'results/{model}-ollama-new.csv'
                conc_level = 1
            else:
                old_file = f'results/{model}-ollama-{conc}.csv'
                new_file = f'results/{model}-ollama-{conc}-new.csv'
                conc_level = int(conc)
            
            # Load old
            if os.path.exists(old_file):
                df_old = pd.read_csv(old_file)
                df_old['version'] = 'Old'
                df_old['model'] = model.capitalize()
                df_old['concurrency'] = conc_level
                data.append(df_old)
            
            # Load new
            if os.path.exists(new_file):
                df_new = pd.read_csv(new_file)
                df_new['version'] = 'New'
                df_new['model'] = model.capitalize()
                df_new['concurrency'] = conc_level
                data.append(df_new)
    
    return pd.concat(data, ignore_index=True)


def create_ttft_comparison(df, output_dir):
    """Create TTFT comparison across concurrency levels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = ['Llama', 'Qwen', 'Gemma']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model]
        
        # Aggregate by concurrency and version
        agg = model_df.groupby(['concurrency', 'version'])['time_to_first_token'].mean().reset_index()
        agg['ttft_ms'] = agg['time_to_first_token'] * 1000
        
        # Pivot for plotting
        pivot = agg.pivot(index='concurrency', columns='version', values='ttft_ms')
        
        x = np.arange(len(pivot.index))
        width = 0.35
        
        bars_old = ax.bar(x - width/2, pivot['Old'], width, label='Old', color=COLORS['old'], alpha=0.8)
        bars_new = ax.bar(x + width/2, pivot['New'], width, label='New', color=COLORS['new'], alpha=0.8)
        
        ax.set_xlabel('Concurrency Level')
        ax.set_ylabel('TTFT (ms)')
        ax.set_title(f'{model}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in pivot.index])
        ax.legend()
        ax.set_yscale('log')
        
        # Add percentage improvement labels
        for i, (old_val, new_val) in enumerate(zip(pivot['Old'], pivot['New'])):
            pct = ((new_val - old_val) / old_val) * 100
            color = COLORS['new'] if pct < 0 else COLORS['old']
            ax.annotate(f'{pct:+.0f}%', xy=(i, max(old_val, new_val)), 
                       ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    fig.suptitle('Time to First Token (TTFT) - Ollama Old vs New', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'01_ttft_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filepath}")


def create_throughput_comparison(df, output_dir):
    """Create throughput comparison across concurrency levels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = ['Llama', 'Qwen', 'Gemma']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model]
        
        # Aggregate by concurrency and version
        agg = model_df.groupby(['concurrency', 'version'])['tokens_per_second'].mean().reset_index()
        
        # Pivot for plotting
        pivot = agg.pivot(index='concurrency', columns='version', values='tokens_per_second')
        
        x = np.arange(len(pivot.index))
        width = 0.35
        
        bars_old = ax.bar(x - width/2, pivot['Old'], width, label='Old', color=COLORS['old'], alpha=0.8)
        bars_new = ax.bar(x + width/2, pivot['New'], width, label='New', color=COLORS['new'], alpha=0.8)
        
        ax.set_xlabel('Concurrency Level')
        ax.set_ylabel('Tokens per Second')
        ax.set_title(f'{model}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in pivot.index])
        ax.legend()
        
        # Add percentage improvement labels
        for i, (old_val, new_val) in enumerate(zip(pivot['Old'], pivot['New'])):
            pct = ((new_val - old_val) / old_val) * 100
            color = COLORS['new'] if pct > 0 else COLORS['old']
            ax.annotate(f'{pct:+.0f}%', xy=(i, max(old_val, new_val)), 
                       ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    fig.suptitle('Throughput (Tokens/Second) - Ollama Old vs New', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'02_throughput_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filepath}")


def create_scalability_lines(df, output_dir):
    """Create line plots showing scalability across concurrency."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = ['Llama', 'Qwen', 'Gemma']
    model_colors = {'Llama': COLORS['llama'], 'Qwen': COLORS['qwen'], 'Gemma': COLORS['gemma']}
    
    # TTFT Old vs New (all models)
    ax1 = axes[0, 0]
    for model in models:
        model_df = df[df['model'] == model]
        for version in ['Old', 'New']:
            version_df = model_df[model_df['version'] == version]
            agg = version_df.groupby('concurrency')['time_to_first_token'].mean() * 1000
            linestyle = '--' if version == 'Old' else '-'
            marker = 'o' if version == 'Old' else 's'
            ax1.plot(agg.index, agg.values, linestyle=linestyle, marker=marker, 
                    label=f'{model} {version}', color=model_colors[model], 
                    alpha=0.6 if version == 'Old' else 1.0, linewidth=2)
    
    ax1.set_xlabel('Concurrency Level')
    ax1.set_ylabel('TTFT (ms)')
    ax1.set_title('TTFT Scalability')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Throughput Old vs New (all models)
    ax2 = axes[0, 1]
    for model in models:
        model_df = df[df['model'] == model]
        for version in ['Old', 'New']:
            version_df = model_df[model_df['version'] == version]
            agg = version_df.groupby('concurrency')['tokens_per_second'].mean()
            linestyle = '--' if version == 'Old' else '-'
            marker = 'o' if version == 'Old' else 's'
            ax2.plot(agg.index, agg.values, linestyle=linestyle, marker=marker,
                    label=f'{model} {version}', color=model_colors[model],
                    alpha=0.6 if version == 'Old' else 1.0, linewidth=2)
    
    ax2.set_xlabel('Concurrency Level')
    ax2.set_ylabel('Tokens per Second')
    ax2.set_title('Throughput Scalability')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # TTFT Improvement % by concurrency
    ax3 = axes[1, 0]
    for model in models:
        model_df = df[df['model'] == model]
        improvements = []
        concurrencies = []
        for conc in sorted(model_df['concurrency'].unique()):
            old_ttft = model_df[(model_df['concurrency'] == conc) & (model_df['version'] == 'Old')]['time_to_first_token'].mean()
            new_ttft = model_df[(model_df['concurrency'] == conc) & (model_df['version'] == 'New')]['time_to_first_token'].mean()
            improvement = ((old_ttft - new_ttft) / old_ttft) * 100
            improvements.append(improvement)
            concurrencies.append(conc)
        ax3.plot(concurrencies, improvements, marker='o', label=model, color=model_colors[model], linewidth=2)
    
    ax3.set_xlabel('Concurrency Level')
    ax3.set_ylabel('TTFT Improvement (%)')
    ax3.set_title('TTFT Improvement (New vs Old)')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Throughput Improvement % by concurrency
    ax4 = axes[1, 1]
    for model in models:
        model_df = df[df['model'] == model]
        improvements = []
        concurrencies = []
        for conc in sorted(model_df['concurrency'].unique()):
            old_tps = model_df[(model_df['concurrency'] == conc) & (model_df['version'] == 'Old')]['tokens_per_second'].mean()
            new_tps = model_df[(model_df['concurrency'] == conc) & (model_df['version'] == 'New')]['tokens_per_second'].mean()
            improvement = ((new_tps - old_tps) / old_tps) * 100
            improvements.append(improvement)
            concurrencies.append(conc)
        ax4.plot(concurrencies, improvements, marker='o', label=model, color=model_colors[model], linewidth=2)
    
    ax4.set_xlabel('Concurrency Level')
    ax4.set_ylabel('Throughput Improvement (%)')
    ax4.set_title('Throughput Improvement (New vs Old)')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Ollama Scalability Analysis - Old vs New', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'03_scalability_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filepath}")


def create_heatmap_comparison(df, output_dir):
    """Create heatmaps comparing Old vs New performance."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    models = ['Llama', 'Qwen', 'Gemma']
    concurrencies = sorted(df['concurrency'].unique())
    
    # TTFT Old
    ax1 = axes[0, 0]
    ttft_old = df[df['version'] == 'Old'].pivot_table(
        index='model', columns='concurrency', values='time_to_first_token', aggfunc='mean'
    ) * 1000
    sns.heatmap(ttft_old, annot=True, fmt='.0f', cmap='Reds', ax=ax1, cbar_kws={'label': 'ms'})
    ax1.set_title('TTFT (ms) - OLD', fontweight='bold')
    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Model')
    
    # TTFT New
    ax2 = axes[0, 1]
    ttft_new = df[df['version'] == 'New'].pivot_table(
        index='model', columns='concurrency', values='time_to_first_token', aggfunc='mean'
    ) * 1000
    sns.heatmap(ttft_new, annot=True, fmt='.0f', cmap='Greens', ax=ax2, cbar_kws={'label': 'ms'})
    ax2.set_title('TTFT (ms) - NEW', fontweight='bold')
    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Model')
    
    # Throughput Old
    ax3 = axes[1, 0]
    tps_old = df[df['version'] == 'Old'].pivot_table(
        index='model', columns='concurrency', values='tokens_per_second', aggfunc='mean'
    )
    sns.heatmap(tps_old, annot=True, fmt='.1f', cmap='Reds', ax=ax3, cbar_kws={'label': 'tok/s'})
    ax3.set_title('Throughput (tok/s) - OLD', fontweight='bold')
    ax3.set_xlabel('Concurrency')
    ax3.set_ylabel('Model')
    
    # Throughput New
    ax4 = axes[1, 1]
    tps_new = df[df['version'] == 'New'].pivot_table(
        index='model', columns='concurrency', values='tokens_per_second', aggfunc='mean'
    )
    sns.heatmap(tps_new, annot=True, fmt='.1f', cmap='Greens', ax=ax4, cbar_kws={'label': 'tok/s'})
    ax4.set_title('Throughput (tok/s) - NEW', fontweight='bold')
    ax4.set_xlabel('Concurrency')
    ax4.set_ylabel('Model')
    
    fig.suptitle('Performance Heatmaps - Ollama Old vs New', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'04_heatmap_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filepath}")


def create_improvement_heatmap(df, output_dir):
    """Create a heatmap showing improvement percentages."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    models = ['Llama', 'Qwen', 'Gemma']
    concurrencies = sorted(df['concurrency'].unique())
    
    # Calculate improvements
    ttft_improvements = pd.DataFrame(index=models, columns=concurrencies)
    tps_improvements = pd.DataFrame(index=models, columns=concurrencies)
    
    for model in models:
        for conc in concurrencies:
            old_df = df[(df['model'] == model) & (df['concurrency'] == conc) & (df['version'] == 'Old')]
            new_df = df[(df['model'] == model) & (df['concurrency'] == conc) & (df['version'] == 'New')]
            
            old_ttft = old_df['time_to_first_token'].mean()
            new_ttft = new_df['time_to_first_token'].mean()
            ttft_improvements.loc[model, conc] = ((old_ttft - new_ttft) / old_ttft) * 100
            
            old_tps = old_df['tokens_per_second'].mean()
            new_tps = new_df['tokens_per_second'].mean()
            tps_improvements.loc[model, conc] = ((new_tps - old_tps) / old_tps) * 100
    
    ttft_improvements = ttft_improvements.astype(float)
    tps_improvements = tps_improvements.astype(float)
    
    # TTFT Improvement (positive = faster, so green)
    ax1 = axes[0]
    sns.heatmap(ttft_improvements, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax1,
                cbar_kws={'label': '% faster'}, vmin=-50, vmax=100)
    ax1.set_title('TTFT Improvement (%)\n(Green = Faster)', fontweight='bold')
    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Model')
    
    # Throughput Improvement (positive = more throughput, so green)
    ax2 = axes[1]
    sns.heatmap(tps_improvements, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax2,
                cbar_kws={'label': '% higher'}, vmin=-50, vmax=400)
    ax2.set_title('Throughput Improvement (%)\n(Green = Higher)', fontweight='bold')
    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Model')
    
    fig.suptitle('Performance Improvement Summary - Ollama New vs Old', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'05_improvement_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filepath}")


def create_latency_distribution(df, output_dir):
    """Create latency distribution comparison."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    models = ['Llama', 'Qwen', 'Gemma']
    
    for idx, model in enumerate(models):
        model_df = df[df['model'] == model]
        
        # TTFT Distribution
        ax1 = axes[idx, 0]
        for version in ['Old', 'New']:
            version_df = model_df[model_df['version'] == version]
            ttft_ms = version_df['time_to_first_token'] * 1000
            color = COLORS['old'] if version == 'Old' else COLORS['new']
            ax1.hist(ttft_ms, bins=50, alpha=0.5, label=version, color=color, density=True)
        ax1.set_xlabel('TTFT (ms)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{model} - TTFT Distribution')
        ax1.legend()
        ax1.set_xscale('log')
        
        # Inter-token latency Distribution
        ax2 = axes[idx, 1]
        for version in ['Old', 'New']:
            version_df = model_df[model_df['version'] == version]
            itl_ms = version_df['avg_inter_token_latency'] * 1000
            color = COLORS['old'] if version == 'Old' else COLORS['new']
            ax2.hist(itl_ms, bins=50, alpha=0.5, label=version, color=color, density=True)
        ax2.set_xlabel('Inter-Token Latency (ms)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'{model} - Inter-Token Latency Distribution')
        ax2.legend()
    
    fig.suptitle('Latency Distributions - Ollama Old vs New', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'06_latency_distributions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filepath}")


def main():
    output_dir = 'viz/ollama_old_vs_new'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🔄 OLLAMA OLD vs NEW - COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    print("\n📂 Loading all Ollama data...")
    df = load_all_ollama_data()
    print(f"   Total records: {len(df)}")
    print(f"   Models: {df['model'].unique().tolist()}")
    print(f"   Concurrency levels: {sorted(df['concurrency'].unique().tolist())}")
    print(f"   Versions: {df['version'].unique().tolist()}")
    
    print("\n📊 Generating visualizations...")
    
    create_ttft_comparison(df, output_dir)
    create_throughput_comparison(df, output_dir)
    create_scalability_lines(df, output_dir)
    create_heatmap_comparison(df, output_dir)
    create_improvement_heatmap(df, output_dir)
    create_latency_distribution(df, output_dir)
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETE! 6 visualizations saved to {output_dir}/")
    print("=" * 70)
    print("\n📁 Generated files:")
    print("   01_ttft_comparison        - TTFT bar charts by model")
    print("   02_throughput_comparison  - Throughput bar charts by model")
    print("   03_scalability_analysis   - Line plots showing scalability")
    print("   04_heatmap_comparison     - Side-by-side heatmaps Old vs New")
    print("   05_improvement_summary    - Improvement % heatmap (key insight!)")
    print("   06_latency_distributions  - Distribution histograms")


if __name__ == "__main__":
    main()
