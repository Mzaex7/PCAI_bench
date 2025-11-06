
"""
Visualization module for benchmark results.
Creates presentation-ready charts comparing model families across inference engines.
Focus: NIM vs vLLM performance comparison with meaningful insights.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Set professional presentation style
sns.set_style("whitegrid")
sns.set_context("talk")  # Larger fonts for presentations
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['figure.facecolor'] = '#FFFFFF'

# HPE Corporate Color Palette
COLORS = {
    # Primary colors for engine comparison
    'nim': '#0070F8',           # HPE Blue - Clear, tech-forward
    'vllm': '#7764FC',          # HPE Purple - Complementary, distinct
    
    # Alternate engine colors (if needed)
    'nim_alt': '#01A982',       # HPE Green - Brand signature
    'vllm_alt': '#62E5F6',      # HPE Cyan - Fresh, modern
    
    # Status indicators
    'success': '#01A982',       # HPE Green - Positive, growth
    'warning': '#05CC93',       # Jade Green - Caution, attention
    'danger': '#7764FC',        # Purple - Issues (softer than red)
    
    # Secondary palette for variety
    'primary': '#01A982',       # HPE Green
    'secondary': '#0070F8',     # Blue
    'accent1': '#05CC93',       # Jade Green
    'accent2': '#00E0AF',       # Mint Green
    'accent3': '#62E5F6',       # Cyan
    'accent4': '#7764FC',       # Purple
    
    # Neutrals for backgrounds and text
    'midnight': '#292D3A',      # Dark text/borders
    'black': '#000000',         # Pure black
    'white': '#FFFFFF',         # Pure white
    'light_cloud': '#F7F7F7',   # Light background
    'cloud': '#E6E8E9',         # Subtle background
    'dark_cloud': '#D4D8DB',    # Medium gray
    'slate': '#B1B9BE',         # Mid gray
    'light_carbon': '#7D8A92',  # Dark gray
    'carbon': '#535C66',        # Very dark gray
}


class BenchmarkVisualizer:
    """Create presentation-ready visualizations for model family comparisons."""
    
    def __init__(self, results: List[Dict], output_dir: str = "visualizations"):
        """
        Initialize visualizer with results.
        
        Args:
            results: List of benchmark result dictionaries
            output_dir: Directory to save visualization files
        """
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        self.df = pd.DataFrame(results)
        
        # Filter successful requests
        self.df_success = self.df[self.df['success'] == True].copy()
        
        # Convert numeric columns
        numeric_cols = [
            'total_latency', 'time_to_first_token', 'time_per_output_token',
            'tokens_per_second', 'input_tokens', 'output_tokens', 'total_tokens',
            'avg_inter_token_latency', 'min_inter_token_latency', 'max_inter_token_latency',
            'p50_inter_token_latency', 'p95_inter_token_latency', 'p99_inter_token_latency'
        ]
        for col in numeric_cols:
            if col in self.df_success.columns:
                self.df_success[col] = pd.to_numeric(self.df_success[col], errors='coerce')
        
        # Extract model family and inference engine from endpoint name
        self._parse_endpoint_info()
    
    def _parse_endpoint_info(self):
        """Extract model family and inference engine from endpoint names."""
        def parse_endpoint(name: str) -> Tuple[str, str]:
            """
            Parse endpoint name to extract model and engine.
            Expected format: 'ModelName-Engine' or 'ModelName_Engine'
            Example: 'Llama3-NIM', 'Mistral-vLLM'
            """
            name_lower = name.lower()
            
            # Detect inference engine
            if 'nim' in name_lower:
                engine = 'NIM'
            elif 'vllm' in name_lower:
                engine = 'vLLM'
            else:
                engine = 'Unknown'
            
            # Extract model family (everything before the engine indicator)
            for separator in ['-nim', '-vllm', '_nim', '_vllm', ' nim', ' vllm']:
                if separator in name_lower:
                    model = name[:name_lower.index(separator)]
                    return model.strip(), engine
            
            # If no clear separator, use the full name
            return name, engine
        
        self.df_success['model_family'] = self.df_success['endpoint_name'].apply(lambda x: parse_endpoint(x)[0])
        self.df_success['inference_engine'] = self.df_success['endpoint_name'].apply(lambda x: parse_endpoint(x)[1])
    
    def _get_engine_color(self, engine: str) -> str:
        """Get color for inference engine."""
        engine_lower = engine.lower()
        if 'nim' in engine_lower:
            return COLORS['nim']
        elif 'vllm' in engine_lower:
            return COLORS['vllm']
        return COLORS['primary']
    
    def _apply_hpe_styling(self, ax):
        """Apply HPE corporate styling to axes."""
        ax.spines['top'].set_color(COLORS['cloud'])
        ax.spines['right'].set_color(COLORS['cloud'])
        ax.spines['bottom'].set_color(COLORS['slate'])
        ax.spines['left'].set_color(COLORS['slate'])
        ax.tick_params(colors=COLORS['midnight'])
        ax.title.set_color(COLORS['midnight'])
        ax.xaxis.label.set_color(COLORS['midnight'])
        ax.yaxis.label.set_color(COLORS['midnight'])
        ax.grid(color=COLORS['cloud'], alpha=0.6)
    
    def _add_value_labels(self, ax, format_str='%.2f', rotation=0):
        """Add value labels on top of bars."""
        for container in ax.containers:
            ax.bar_label(container, fmt=format_str, padding=3, rotation=rotation)
    
    def _save_figure(self, name: str, dpi: int = 300) -> str:
        """Save figure with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filepath}")
        plt.close()
        return filepath
    
    # ============================================================================
    # KEY PERFORMANCE METRICS - Side-by-side engine comparison
    # ============================================================================
    
    def plot_engine_comparison_dashboard(self, save: bool = True) -> str:
        """
        Create a comprehensive dashboard comparing NIM vs vLLM across key metrics.
        Perfect for executive summary slide.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Inference Engine Performance Comparison: NIM vs vLLM', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        # Group by model family and engine
        grouped = self.df_success.groupby(['model_family', 'inference_engine']).agg({
            'time_to_first_token': 'mean',
            'tokens_per_second': 'mean',
            'avg_inter_token_latency': 'mean',
            'total_latency': 'mean'
        }).reset_index()
        
        # 1. TTFT Comparison
        ax1 = axes[0, 0]
        pivot_ttft = grouped.pivot(index='model_family', columns='inference_engine', values='time_to_first_token')
        pivot_ttft.plot(kind='bar', ax=ax1, color=[self._get_engine_color(col) for col in pivot_ttft.columns], width=0.7)
        ax1.set_title('Time to First Token (TTFT)\nLower is Better', fontweight='bold', pad=10)
        ax1.set_ylabel('Seconds', fontweight='bold')
        ax1.set_xlabel('Model Family', fontweight='bold')
        ax1.legend(title='Engine', frameon=True, shadow=True)
        ax1.grid(axis='y', alpha=0.6, color=COLORS['cloud'])
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Throughput Comparison
        ax2 = axes[0, 1]
        pivot_tps = grouped.pivot(index='model_family', columns='inference_engine', values='tokens_per_second')
        pivot_tps.plot(kind='bar', ax=ax2, color=[self._get_engine_color(col) for col in pivot_tps.columns], width=0.7)
        ax2.set_title('Throughput (Tokens/Second)\nHigher is Better', fontweight='bold', pad=10)
        ax2.set_ylabel('Tokens/Second', fontweight='bold')
        ax2.set_xlabel('Model Family', fontweight='bold')
        ax2.legend(title='Engine', frameon=True, shadow=True)
        ax2.grid(axis='y', alpha=0.6, color=COLORS['cloud'])
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Inter-Token Latency
        ax3 = axes[1, 0]
        df_itl = grouped.dropna(subset=['avg_inter_token_latency'])
        if len(df_itl) > 0:
            pivot_itl = df_itl.pivot(index='model_family', columns='inference_engine', values='avg_inter_token_latency')
            pivot_itl.plot(kind='bar', ax=ax3, color=[self._get_engine_color(col) for col in pivot_itl.columns], width=0.7)
            ax3.set_title('Average Inter-Token Latency\nLower is Better', fontweight='bold', pad=10)
            ax3.set_ylabel('Seconds', fontweight='bold')
            ax3.set_xlabel('Model Family', fontweight='bold')
            ax3.legend(title='Engine', frameon=True, shadow=True)
            ax3.grid(axis='y', alpha=0.6, color=COLORS['cloud'])
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Total Latency
        ax4 = axes[1, 1]
        pivot_latency = grouped.pivot(index='model_family', columns='inference_engine', values='total_latency')
        pivot_latency.plot(kind='bar', ax=ax4, color=[self._get_engine_color(col) for col in pivot_latency.columns], width=0.7)
        ax4.set_title('Total Request Latency\nLower is Better', fontweight='bold', pad=10)
        ax4.set_ylabel('Seconds', fontweight='bold')
        ax4.set_xlabel('Model Family', fontweight='bold')
        ax4.legend(title='Engine', frameon=True, shadow=True)
        ax4.grid(axis='y', alpha=0.6, color=COLORS['cloud'])
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('01_engine_comparison_dashboard')
        return None
    
    def plot_ttft_detailed_comparison(self, save: bool = True) -> str:
        """
        Detailed TTFT comparison with distribution analysis.
        Shows responsiveness - critical for user experience.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Time to First Token (TTFT) Analysis - Responsiveness Metric', 
                     fontsize=16, fontweight='bold')
        
        df_ttft = self.df_success.dropna(subset=['time_to_first_token'])
        
        # 1. Box plot showing distribution
        ax1 = axes[0]
        engines = sorted(df_ttft['inference_engine'].unique())
        colors = [self._get_engine_color(eng) for eng in engines]
        
        box_parts = ax1.boxplot(
            [df_ttft[df_ttft['inference_engine'] == eng]['time_to_first_token'] for eng in engines],
            labels=engines,
            patch_artist=True,
            showmeans=True,
            meanline=True
        )
        
        for patch, color in zip(box_parts['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('TTFT Distribution by Engine', fontweight='bold', pad=10)
        ax1.set_ylabel('TTFT (seconds)', fontweight='bold')
        ax1.set_xlabel('Inference Engine', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add statistical annotations
        for i, eng in enumerate(engines, 1):
            data = df_ttft[df_ttft['inference_engine'] == eng]['time_to_first_token']
            median = data.median()
            ax1.text(i, median, f'{median:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Grouped bar chart by model family
        ax2 = axes[1]
        grouped = df_ttft.groupby(['model_family', 'inference_engine'])['time_to_first_token'].mean().reset_index()
        pivot = grouped.pivot(index='model_family', columns='inference_engine', values='time_to_first_token')
        
        pivot.plot(kind='bar', ax=ax2, color=[self._get_engine_color(col) for col in pivot.columns], width=0.7)
        ax2.set_title('Average TTFT by Model Family', fontweight='bold', pad=10)
        ax2.set_ylabel('TTFT (seconds)', fontweight='bold')
        ax2.set_xlabel('Model Family', fontweight='bold')
        ax2.legend(title='Engine', frameon=True, shadow=True)
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add percentage difference annotations
        for idx, model in enumerate(pivot.index):
            if len(pivot.columns) == 2:
                val1, val2 = pivot.iloc[idx]
                if pd.notna(val1) and pd.notna(val2):
                    diff_pct = ((val2 - val1) / val1) * 100
                    color = COLORS['success'] if diff_pct < 0 else COLORS['danger']
                    ax2.text(idx, max(val1, val2) * 1.05, f'{diff_pct:+.1f}%', 
                            ha='center', fontsize=9, color=color, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('02_ttft_detailed_comparison')
        return None
    
    def plot_throughput_analysis(self, save: bool = True) -> str:
        """
        Comprehensive throughput analysis - key metric for production capacity.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        fig.suptitle('Throughput Analysis - Production Capacity Metric', 
                     fontsize=16, fontweight='bold')
        
        df_tps = self.df_success.dropna(subset=['tokens_per_second'])
        
        # 1. Average throughput by engine (large bar chart)
        ax1 = fig.add_subplot(gs[0, :])
        grouped = df_tps.groupby(['model_family', 'inference_engine'])['tokens_per_second'].mean().reset_index()
        pivot = grouped.pivot(index='model_family', columns='inference_engine', values='tokens_per_second')
        
        pivot.plot(kind='bar', ax=ax1, color=[self._get_engine_color(col) for col in pivot.columns], width=0.7)
        ax1.set_title('Average Throughput by Model Family and Engine', fontweight='bold', pad=10, fontsize=14)
        ax1.set_ylabel('Tokens per Second', fontweight='bold')
        ax1.set_xlabel('Model Family', fontweight='bold')
        ax1.legend(title='Engine', frameon=True, shadow=True, fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
        
        # 2. Violin plot - distribution comparison
        ax2 = fig.add_subplot(gs[1, 0])
        engines = sorted(df_tps['inference_engine'].unique())
        positions = range(len(engines))
        
        violin_parts = ax2.violinplot(
            [df_tps[df_tps['inference_engine'] == eng]['tokens_per_second'] for eng in engines],
            positions=positions,
            showmeans=True,
            showmedians=True
        )
        
        for i, (pc, eng) in enumerate(zip(violin_parts['bodies'], engines)):
            pc.set_facecolor(self._get_engine_color(eng))
            pc.set_alpha(0.7)
        
        ax2.set_xticks(positions)
        ax2.set_xticklabels(engines)
        ax2.set_title('Throughput Distribution', fontweight='bold', pad=10)
        ax2.set_ylabel('Tokens per Second', fontweight='bold')
        ax2.set_xlabel('Inference Engine', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Performance by prompt length
        ax3 = fig.add_subplot(gs[1, 1])
        if 'prompt_length' in df_tps.columns:
            length_grouped = df_tps.groupby(['prompt_length', 'inference_engine'])['tokens_per_second'].mean().reset_index()
            
            for engine in engines:
                engine_data = length_grouped[length_grouped['inference_engine'] == engine]
                ax3.plot(engine_data['prompt_length'], engine_data['tokens_per_second'], 
                        marker='o', linewidth=2, markersize=8, label=engine,
                        color=self._get_engine_color(engine))
            
            ax3.set_title('Throughput vs Prompt Length', fontweight='bold', pad=10)
            ax3.set_ylabel('Tokens per Second', fontweight='bold')
            ax3.set_xlabel('Prompt Length', fontweight='bold')
            ax3.legend(title='Engine', frameon=True, shadow=True)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('03_throughput_analysis')
        return None
    
    def plot_inter_token_latency_analysis(self, save: bool = True) -> str:
        """
        Analyze inter-token latency - critical for streaming applications.
        Shows consistency and predictability of token generation.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Inter-Token Latency Analysis - Streaming Consistency', 
                     fontsize=16, fontweight='bold')
        
        df_itl = self.df_success.dropna(subset=['avg_inter_token_latency'])
        
        if len(df_itl) == 0:
            print("⚠ No inter-token latency data available")
            plt.close()
            return None
        
        # 1. Average ITL comparison
        ax1 = axes[0, 0]
        grouped = df_itl.groupby(['model_family', 'inference_engine'])['avg_inter_token_latency'].mean().reset_index()
        pivot = grouped.pivot(index='model_family', columns='inference_engine', values='avg_inter_token_latency')
        
        pivot.plot(kind='bar', ax=ax1, color=[self._get_engine_color(col) for col in pivot.columns], width=0.7)
        ax1.set_title('Average Inter-Token Latency', fontweight='bold', pad=10)
        ax1.set_ylabel('Seconds', fontweight='bold')
        ax1.set_xlabel('Model Family', fontweight='bold')
        ax1.legend(title='Engine', frameon=True, shadow=True)
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Consistency analysis (min vs max)
        ax2 = axes[0, 1]
        df_consistency = df_itl.groupby(['model_family', 'inference_engine']).agg({
            'min_inter_token_latency': 'mean',
            'max_inter_token_latency': 'mean'
        }).reset_index()
        
        x = np.arange(len(df_consistency['model_family'].unique()))
        width = 0.35
        engines = df_consistency['inference_engine'].unique()
        
        for i, engine in enumerate(engines):
            engine_data = df_consistency[df_consistency['inference_engine'] == engine]
            offset = width * (i - 0.5)
            ax2.bar(x + offset, engine_data['max_inter_token_latency'] - engine_data['min_inter_token_latency'],
                   width, label=f'{engine} (Range)', color=self._get_engine_color(engine), alpha=0.7)
        
        ax2.set_title('ITL Consistency (Max - Min Range)', fontweight='bold', pad=10)
        ax2.set_ylabel('Latency Range (seconds)', fontweight='bold')
        ax2.set_xlabel('Model Family', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_consistency['model_family'].unique(), rotation=45, ha='right')
        ax2.legend(frameon=True, shadow=True)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Percentile comparison (P50, P95, P99)
        ax3 = axes[1, 0]
        percentile_data = df_itl.groupby(['inference_engine']).agg({
            'p50_inter_token_latency': 'mean',
            'p95_inter_token_latency': 'mean',
            'p99_inter_token_latency': 'mean'
        }).reset_index()
        
        x = np.arange(len(percentile_data))
        width = 0.25
        
        p50 = ax3.bar(x - width, percentile_data['p50_inter_token_latency'], width, 
                     label='P50 (Median)', color=COLORS['success'], alpha=0.8)
        p95 = ax3.bar(x, percentile_data['p95_inter_token_latency'], width,
                     label='P95', color=COLORS['warning'], alpha=0.8)
        p99 = ax3.bar(x + width, percentile_data['p99_inter_token_latency'], width,
                     label='P99 (Worst Case)', color=COLORS['danger'], alpha=0.8)
        
        ax3.set_title('ITL Percentiles by Engine', fontweight='bold', pad=10)
        ax3.set_ylabel('Latency (seconds)', fontweight='bold')
        ax3.set_xlabel('Inference Engine', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(percentile_data['inference_engine'])
        ax3.legend(frameon=True, shadow=True)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [p50, p95, p99]:
            ax3.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        
        # 4. Box plot comparison
        ax4 = axes[1, 1]
        engines = sorted(df_itl['inference_engine'].unique())
        box_parts = ax4.boxplot(
            [df_itl[df_itl['inference_engine'] == eng]['avg_inter_token_latency'] for eng in engines],
            labels=engines,
            patch_artist=True,
            showmeans=True,
            meanline=True
        )
        
        for patch, eng in zip(box_parts['boxes'], engines):
            patch.set_facecolor(self._get_engine_color(eng))
            patch.set_alpha(0.7)
        
        ax4.set_title('ITL Distribution by Engine', fontweight='bold', pad=10)
        ax4.set_ylabel('Average ITL (seconds)', fontweight='bold')
        ax4.set_xlabel('Inference Engine', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('04_inter_token_latency_analysis')
        return None
    
    def plot_scalability_analysis(self, save: bool = True) -> str:
        """
        Analyze performance across different prompt lengths and complexities.
        Shows how engines scale with workload.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scalability Analysis - Performance Under Different Workloads', 
                     fontsize=16, fontweight='bold')
        
        # 1. Latency vs Prompt Length
        ax1 = axes[0, 0]
        if 'prompt_length' in self.df_success.columns:
            length_grouped = self.df_success.groupby(['prompt_length', 'inference_engine'])['total_latency'].mean().reset_index()
            engines = sorted(self.df_success['inference_engine'].unique())
            
            for engine in engines:
                engine_data = length_grouped[length_grouped['inference_engine'] == engine]
                ax1.plot(engine_data['prompt_length'], engine_data['total_latency'], 
                        marker='o', linewidth=2.5, markersize=10, label=engine,
                        color=self._get_engine_color(engine))
            
            ax1.set_title('Total Latency vs Prompt Length', fontweight='bold', pad=10)
            ax1.set_ylabel('Latency (seconds)', fontweight='bold')
            ax1.set_xlabel('Prompt Length', fontweight='bold')
            ax1.legend(title='Engine', frameon=True, shadow=True)
            ax1.grid(True, alpha=0.3)
        
        # 2. Throughput vs Prompt Complexity
        ax2 = axes[0, 1]
        if 'prompt_complexity' in self.df_success.columns:
            complexity_grouped = self.df_success.dropna(subset=['tokens_per_second']).groupby(
                ['prompt_complexity', 'inference_engine'])['tokens_per_second'].mean().reset_index()
            
            pivot = complexity_grouped.pivot(index='prompt_complexity', columns='inference_engine', values='tokens_per_second')
            pivot.plot(kind='bar', ax=ax2, color=[self._get_engine_color(col) for col in pivot.columns], width=0.7)
            
            ax2.set_title('Throughput vs Prompt Complexity', fontweight='bold', pad=10)
            ax2.set_ylabel('Tokens per Second', fontweight='bold')
            ax2.set_xlabel('Prompt Complexity', fontweight='bold')
            ax2.legend(title='Engine', frameon=True, shadow=True)
            ax2.grid(axis='y', alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. TTFT scaling
        ax3 = axes[1, 0]
        if 'prompt_length' in self.df_success.columns:
            ttft_grouped = self.df_success.dropna(subset=['time_to_first_token']).groupby(
                ['prompt_length', 'inference_engine'])['time_to_first_token'].mean().reset_index()
            
            for engine in engines:
                engine_data = ttft_grouped[ttft_grouped['inference_engine'] == engine]
                ax3.plot(engine_data['prompt_length'], engine_data['time_to_first_token'], 
                        marker='s', linewidth=2.5, markersize=10, label=engine,
                        color=self._get_engine_color(engine))
            
            ax3.set_title('TTFT Scaling with Prompt Length', fontweight='bold', pad=10)
            ax3.set_ylabel('TTFT (seconds)', fontweight='bold')
            ax3.set_xlabel('Prompt Length', fontweight='bold')
            ax3.legend(title='Engine', frameon=True, shadow=True)
            ax3.grid(True, alpha=0.3)
        
        # 4. Output tokens vs latency
        ax4 = axes[1, 1]
        df_tokens = self.df_success.dropna(subset=['output_tokens', 'total_latency'])
        
        for engine in engines:
            engine_data = df_tokens[df_tokens['inference_engine'] == engine]
            ax4.scatter(engine_data['output_tokens'], engine_data['total_latency'],
                       alpha=0.6, s=80, label=engine, color=self._get_engine_color(engine))
        
        ax4.set_title('Latency vs Output Tokens', fontweight='bold', pad=10)
        ax4.set_ylabel('Total Latency (seconds)', fontweight='bold')
        ax4.set_xlabel('Output Tokens', fontweight='bold')
        ax4.legend(title='Engine', frameon=True, shadow=True)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('05_scalability_analysis')
        return None
    
    def plot_performance_heatmap(self, save: bool = True) -> str:
        """
        Create heatmap showing relative performance across multiple dimensions.
        Excellent for identifying patterns and optimal configurations.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Performance Heatmap - Comprehensive Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Prepare data for heatmap
        metrics = ['time_to_first_token', 'avg_inter_token_latency', 'tokens_per_second', 'total_latency']
        available_metrics = [m for m in metrics if m in self.df_success.columns and self.df_success[m].notna().any()]
        
        if len(available_metrics) == 0:
            print("⚠ No metrics available for heatmap")
            plt.close()
            return None
        
        # 1. Engine comparison heatmap
        ax1 = axes[0]
        engine_perf = self.df_success.groupby('inference_engine')[available_metrics].mean()
        
        # Normalize each metric (0-1 scale, inverted for latency metrics)
        normalized = engine_perf.copy()
        for col in normalized.columns:
            if 'latency' in col.lower():
                # Lower is better - invert
                normalized[col] = 1 - (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
            else:
                # Higher is better
                normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
        
        # Create custom HPE colormap (white -> HPE Green)
        from matplotlib.colors import LinearSegmentedColormap
        hpe_colors = ['#FFFFFF', '#E6E8E9', '#00E0AF', '#05CC93', '#01A982']
        n_bins = 100
        hpe_cmap = LinearSegmentedColormap.from_list('hpe', hpe_colors, N=n_bins)
        
        sns.heatmap(normalized.T, annot=True, fmt='.3f', cmap=hpe_cmap, 
                   center=0.5, vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'Normalized Score (0-1)'},
                   linewidths=0.5, linecolor=COLORS['cloud'])
        ax1.set_title('Engine Performance Score\n(1.0 = Best, 0.0 = Worst)', fontweight='bold', pad=10)
        ax1.set_ylabel('Metric', fontweight='bold')
        ax1.set_xlabel('Inference Engine', fontweight='bold')
        
        # 2. Model family heatmap
        ax2 = axes[1]
        family_perf = self.df_success.groupby('model_family')[available_metrics].mean()
        
        # Normalize
        normalized_family = family_perf.copy()
        for col in normalized_family.columns:
            if 'latency' in col.lower():
                normalized_family[col] = 1 - (normalized_family[col] - normalized_family[col].min()) / (normalized_family[col].max() - normalized_family[col].min())
            else:
                normalized_family[col] = (normalized_family[col] - normalized_family[col].min()) / (normalized_family[col].max() - normalized_family[col].min())
        
        sns.heatmap(normalized_family.T, annot=True, fmt='.3f', cmap=hpe_cmap,
                   center=0.5, vmin=0, vmax=1, ax=ax2, cbar_kws={'label': 'Normalized Score (0-1)'},
                   linewidths=0.5, linecolor=COLORS['cloud'])
        ax2.set_title('Model Family Performance Score\n(1.0 = Best, 0.0 = Worst)', fontweight='bold', pad=10)
        ax2.set_ylabel('Metric', fontweight='bold')
        ax2.set_xlabel('Model Family', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('06_performance_heatmap')
        return None
    
    def plot_cost_efficiency_analysis(self, save: bool = True) -> str:
        """
        Analyze efficiency: throughput vs latency trade-off.
        Helps identify the sweet spot for production deployment.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Efficiency Analysis - Performance Trade-offs', 
                     fontsize=16, fontweight='bold')
        
        df_efficiency = self.df_success.dropna(subset=['tokens_per_second', 'total_latency'])
        
        # 1. Throughput vs Latency scatter
        ax1 = axes[0]
        engines = sorted(df_efficiency['inference_engine'].unique())
        
        for engine in engines:
            engine_data = df_efficiency[df_efficiency['inference_engine'] == engine]
            ax1.scatter(engine_data['total_latency'], engine_data['tokens_per_second'],
                       alpha=0.6, s=100, label=engine, color=self._get_engine_color(engine))
            
            # Add average point
            avg_latency = engine_data['total_latency'].mean()
            avg_throughput = engine_data['tokens_per_second'].mean()
            ax1.scatter(avg_latency, avg_throughput, s=300, marker='*',
                       edgecolors='black', linewidths=2, color=self._get_engine_color(engine),
                       label=f'{engine} (avg)')
        
        ax1.set_title('Throughput vs Latency Trade-off', fontweight='bold', pad=10)
        ax1.set_xlabel('Total Latency (seconds) - Lower is Better', fontweight='bold')
        ax1.set_ylabel('Throughput (tokens/s) - Higher is Better', fontweight='bold')
        ax1.legend(frameon=True, shadow=True, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add quadrant lines
        median_latency = df_efficiency['total_latency'].median()
        median_throughput = df_efficiency['tokens_per_second'].median()
        ax1.axvline(median_latency, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(median_throughput, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Label quadrants
        ax1.text(ax1.get_xlim()[0] * 1.05, ax1.get_ylim()[1] * 0.95, 'IDEAL\n(Low Latency\nHigh Throughput)', 
                fontsize=9, style='italic', alpha=0.5, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))
        
        # 2. Efficiency score by model family
        ax2 = axes[1]
        
        # Calculate efficiency score: throughput / latency (higher is better)
        grouped = df_efficiency.groupby(['model_family', 'inference_engine']).agg({
            'tokens_per_second': 'mean',
            'total_latency': 'mean'
        }).reset_index()
        grouped['efficiency_score'] = grouped['tokens_per_second'] / grouped['total_latency']
        
        pivot = grouped.pivot(index='model_family', columns='inference_engine', values='efficiency_score')
        pivot.plot(kind='bar', ax=ax2, color=[self._get_engine_color(col) for col in pivot.columns], width=0.7)
        
        ax2.set_title('Efficiency Score by Model Family\n(Throughput/Latency Ratio)', fontweight='bold', pad=10)
        ax2.set_ylabel('Efficiency Score (higher is better)', fontweight='bold')
        ax2.set_xlabel('Model Family', fontweight='bold')
        ax2.legend(title='Engine', frameon=True, shadow=True)
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('07_efficiency_analysis')
        return None
    
    def plot_reliability_metrics(self, save: bool = True) -> str:
        """
        Analyze reliability: success rates, error patterns, and consistency.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Reliability Metrics - Production Readiness', 
                     fontsize=16, fontweight='bold')
        
        # Parse endpoint info if not already done
        if 'inference_engine' not in self.df.columns:
            def parse_endpoint(name: str) -> Tuple[str, str]:
                name_lower = name.lower()
                if 'nim' in name_lower:
                    engine = 'NIM'
                elif 'vllm' in name_lower:
                    engine = 'vLLM'
                else:
                    engine = 'Unknown'
                
                for separator in ['-nim', '-vllm', '_nim', '_vllm', ' nim', ' vllm']:
                    if separator in name_lower:
                        model = name[:name_lower.index(separator)]
                        return model.strip(), engine
                
                return name, engine
            
            self.df['model_family'] = self.df['endpoint_name'].apply(lambda x: parse_endpoint(x)[0])
            self.df['inference_engine'] = self.df['endpoint_name'].apply(lambda x: parse_endpoint(x)[1])
        
        # 1. Success rate by engine
        ax1 = axes[0, 0]
        success_stats = self.df.groupby(['inference_engine', 'success']).size().unstack(fill_value=0)
        success_rates = (success_stats[True] / success_stats.sum(axis=1) * 100).reset_index()
        success_rates.columns = ['inference_engine', 'success_rate']
        
        bars = ax1.bar(range(len(success_rates)), success_rates['success_rate'],
                      color=[self._get_engine_color(eng) for eng in success_rates['inference_engine']],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(success_rates)))
        ax1.set_xticklabels(success_rates['inference_engine'])
        ax1.set_title('Success Rate by Engine', fontweight='bold', pad=10)
        ax1.set_ylabel('Success Rate (%)', fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels and color coding
        for i, (bar, rate) in enumerate(zip(bars, success_rates['success_rate'])):
            color = COLORS['success'] if rate >= 95 else COLORS['warning'] if rate >= 90 else COLORS['danger']
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', fontweight='bold', color=color, fontsize=12)
        
        # 2. Success rate by model family
        ax2 = axes[0, 1]
        family_success = self.df.groupby(['model_family', 'success']).size().unstack(fill_value=0)
        family_rates = (family_success[True] / family_success.sum(axis=1) * 100).reset_index()
        family_rates.columns = ['model_family', 'success_rate']
        
        bars = ax2.barh(range(len(family_rates)), family_rates['success_rate'],
                       color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(range(len(family_rates)))
        ax2.set_yticklabels(family_rates['model_family'])
        ax2.set_title('Success Rate by Model Family', fontweight='bold', pad=10)
        ax2.set_xlabel('Success Rate (%)', fontweight='bold')
        ax2.set_xlim(0, 105)
        ax2.grid(axis='x', alpha=0.3)
        
        for bar, rate in zip(bars, family_rates['success_rate']):
            ax2.text(rate + 1, bar.get_y() + bar.get_height()/2, 
                    f'{rate:.1f}%', va='center', fontweight='bold', fontsize=10)
        
        # 3. Latency variance (coefficient of variation)
        ax3 = axes[1, 0]
        variance_data = self.df_success.groupby('inference_engine')['total_latency'].agg(['mean', 'std']).reset_index()
        variance_data['cv'] = (variance_data['std'] / variance_data['mean']) * 100  # Coefficient of variation
        
        bars = ax3.bar(range(len(variance_data)), variance_data['cv'],
                      color=[self._get_engine_color(eng) for eng in variance_data['inference_engine']],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(variance_data)))
        ax3.set_xticklabels(variance_data['inference_engine'])
        ax3.set_title('Latency Consistency (Coefficient of Variation)\nLower = More Consistent', 
                     fontweight='bold', pad=10)
        ax3.set_ylabel('CV (%)', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, cv in zip(bars, variance_data['cv']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{cv:.1f}%', ha='center', fontweight='bold', fontsize=10)
        
        # 4. Error distribution (if any failures)
        ax4 = axes[1, 1]
        failures = self.df[self.df['success'] == False]
        
        if len(failures) > 0:
            error_counts = failures.groupby(['inference_engine', 'error_message']).size().reset_index(name='count')
            
            # Create grouped bar chart
            engines = error_counts['inference_engine'].unique()
            error_types = error_counts['error_message'].unique()
            
            x = np.arange(len(engines))
            width = 0.8 / len(error_types)
            
            for i, error_type in enumerate(error_types):
                error_data = error_counts[error_counts['error_message'] == error_type]
                counts = [error_data[error_data['inference_engine'] == eng]['count'].sum() 
                         if eng in error_data['inference_engine'].values else 0 for eng in engines]
                ax4.bar(x + i * width, counts, width, label=error_type[:30], alpha=0.8)
            
            ax4.set_xticks(x + width * (len(error_types) - 1) / 2)
            ax4.set_xticklabels(engines)
            ax4.set_title('Error Distribution by Type', fontweight='bold', pad=10)
            ax4.set_ylabel('Error Count', fontweight='bold')
            ax4.legend(frameon=True, shadow=True, fontsize=8)
            ax4.grid(axis='y', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '✓ No Errors Detected\n100% Success Rate', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color=COLORS['success'], transform=ax4.transAxes)
            ax4.set_title('Error Distribution', fontweight='bold', pad=10)
            ax4.axis('off')
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('08_reliability_metrics')
        return None
    
    def plot_executive_summary(self, save: bool = True) -> str:
        """
        Create a single-page executive summary with key insights.
        Perfect for presentations and stakeholder reviews.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        fig.suptitle('Executive Summary - NIM vs vLLM Performance Benchmark', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # Calculate key metrics
        grouped = self.df_success.groupby('inference_engine').agg({
            'time_to_first_token': ['mean', 'std'],
            'tokens_per_second': ['mean', 'std'],
            'total_latency': ['mean', 'std'],
            'avg_inter_token_latency': 'mean'
        }).round(3)
        
        # Parse endpoint info for full df if not already done
        if 'inference_engine' not in self.df.columns:
            def parse_endpoint(name: str) -> Tuple[str, str]:
                name_lower = name.lower()
                if 'nim' in name_lower:
                    engine = 'NIM'
                elif 'vllm' in name_lower:
                    engine = 'vLLM'
                else:
                    engine = 'Unknown'
                
                for separator in ['-nim', '-vllm', '_nim', '_vllm', ' nim', ' vllm']:
                    if separator in name_lower:
                        model = name[:name_lower.index(separator)]
                        return model.strip(), engine
                
                return name, engine
            
            self.df['inference_engine'] = self.df['endpoint_name'].apply(lambda x: parse_endpoint(x)[1])
        
        success_stats = self.df.groupby('inference_engine')['success'].agg(['sum', 'count'])
        success_rates = (success_stats['sum'] / success_stats['count'] * 100).round(1)
        
        # 1. Key Metrics Cards (Top Row)
        engines = sorted(self.df_success['inference_engine'].unique())
        
        for i, engine in enumerate(engines):
            ax = fig.add_subplot(gs[0, i])
            ax.axis('off')
            
            ttft = grouped.loc[engine, ('time_to_first_token', 'mean')]
            tps = grouped.loc[engine, ('tokens_per_second', 'mean')]
            latency = grouped.loc[engine, ('total_latency', 'mean')]
            success = success_rates[engine]
            
            # Create card
            card_color = self._get_engine_color(engine)
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=card_color, alpha=0.1, edgecolor=card_color, linewidth=3))
            
            ax.text(0.5, 0.85, engine, ha='center', va='top', fontsize=18, fontweight='bold', color=card_color)
            ax.text(0.5, 0.65, f'TTFT: {ttft:.3f}s', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.50, f'Throughput: {tps:.1f} t/s', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.35, f'Latency: {latency:.3f}s', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.15, f'Success: {success:.1f}%', ha='center', va='center', fontsize=12, 
                   fontweight='bold', color=COLORS['success'] if success >= 95 else COLORS['warning'])
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Winner card
        ax_winner = fig.add_subplot(gs[0, 2])
        ax_winner.axis('off')
        
        # Determine winner
        ttft_winner = grouped[('time_to_first_token', 'mean')].idxmin()
        tps_winner = grouped[('tokens_per_second', 'mean')].idxmax()
        
        ax_winner.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['success'], alpha=0.15, 
                                         edgecolor=COLORS['success'], linewidth=3))
        ax_winner.text(0.5, 0.85, '🏆 Winners', ha='center', va='top', fontsize=18, 
                      fontweight='bold', color=COLORS['midnight'])
        ax_winner.text(0.5, 0.60, f'Fastest TTFT:\n{ttft_winner}', ha='center', va='center', 
                      fontsize=12, fontweight='bold', color=COLORS['midnight'])
        ax_winner.text(0.5, 0.30, f'Best Throughput:\n{tps_winner}', ha='center', va='center', 
                      fontsize=12, fontweight='bold', color=COLORS['midnight'])
        ax_winner.set_xlim(0, 1)
        ax_winner.set_ylim(0, 1)
        
        # 2. Performance Comparison (Middle Row)
        ax_perf = fig.add_subplot(gs[1, :2])
        metrics_to_plot = ['time_to_first_token', 'tokens_per_second', 'total_latency']
        x = np.arange(len(engines))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = [grouped.loc[eng, (metric, 'mean')] for eng in engines]
            
            # Normalize for better visualization
            if metric == 'tokens_per_second':
                ax_perf.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(),
                           alpha=0.8, edgecolor='black', linewidth=1)
            else:
                # Invert latency metrics for visual consistency (lower is better)
                max_val = max(values)
                inverted = [max_val - v for v in values]
                ax_perf.bar(x + i*width, inverted, width, label=f'{metric.replace("_", " ").title()} (inverted)',
                           alpha=0.8, edgecolor='black', linewidth=1)
        
        ax_perf.set_xticks(x + width)
        ax_perf.set_xticklabels(engines)
        ax_perf.set_title('Key Performance Metrics Comparison', fontweight='bold', pad=10, fontsize=14)
        ax_perf.set_ylabel('Performance Score', fontweight='bold')
        ax_perf.legend(frameon=True, shadow=True)
        ax_perf.grid(axis='y', alpha=0.3)
        
        # 3. Inter-Token Latency Percentiles (Middle Right)
        ax_itl = fig.add_subplot(gs[1, 2])
        df_itl = self.df_success.dropna(subset=['p50_inter_token_latency', 'p95_inter_token_latency', 'p99_inter_token_latency'])
        
        if len(df_itl) > 0:
            itl_data = df_itl.groupby('inference_engine').agg({
                'p50_inter_token_latency': 'mean',
                'p95_inter_token_latency': 'mean',
                'p99_inter_token_latency': 'mean'
            }).reset_index()
            
            x = np.arange(len(itl_data))
            width = 0.25
            
            ax_itl.bar(x - width, itl_data['p50_inter_token_latency'], width, label='P50', color=COLORS['success'], alpha=0.8)
            ax_itl.bar(x, itl_data['p95_inter_token_latency'], width, label='P95', color=COLORS['warning'], alpha=0.8)
            ax_itl.bar(x + width, itl_data['p99_inter_token_latency'], width, label='P99', color=COLORS['danger'], alpha=0.8)
            
            ax_itl.set_xticks(x)
            ax_itl.set_xticklabels(itl_data['inference_engine'])
            ax_itl.set_title('Inter-Token Latency\nPercentiles', fontweight='bold', pad=10)
            ax_itl.set_ylabel('Seconds', fontweight='bold')
            ax_itl.legend(frameon=True, shadow=True, fontsize=9)
            ax_itl.grid(axis='y', alpha=0.3)
        
        # 4. Model Family Comparison (Bottom)
        ax_family = fig.add_subplot(gs[2, :])
        family_grouped = self.df_success.groupby(['model_family', 'inference_engine']).agg({
            'tokens_per_second': 'mean'
        }).reset_index()
        
        pivot = family_grouped.pivot(index='model_family', columns='inference_engine', values='tokens_per_second')
        pivot.plot(kind='bar', ax=ax_family, color=[self._get_engine_color(col) for col in pivot.columns], width=0.7)
        
        ax_family.set_title('Throughput by Model Family', fontweight='bold', pad=10, fontsize=14)
        ax_family.set_ylabel('Tokens per Second', fontweight='bold')
        ax_family.set_xlabel('Model Family', fontweight='bold')
        ax_family.legend(title='Engine', frameon=True, shadow=True)
        ax_family.grid(axis='y', alpha=0.3)
        plt.setp(ax_family.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for container in ax_family.containers:
            ax_family.bar_label(container, fmt='%.1f', padding=3, fontsize=8)
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('00_executive_summary', dpi=150)  # Lower DPI for faster generation
        return None
    
    def generate_all_visualizations(self) -> List[str]:
        """
        Generate all presentation-ready visualizations.
        Ordered for logical flow in presentations.
        
        Returns:
            List of file paths for all generated visualizations
        """
        print(f"\n{'='*80}")
        print("🎨 GENERATING PRESENTATION-READY VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        if len(self.df_success) == 0:
            print("⚠ No successful results to visualize")
            return []
        
        print(f"Data Summary:")
        print(f"  • Total requests: {len(self.df)}")
        print(f"  • Successful requests: {len(self.df_success)}")
        print(f"  • Inference engines: {', '.join(self.df_success['inference_engine'].unique())}")
        print(f"  • Model families: {', '.join(self.df_success['model_family'].unique())}")
        print(f"\nGenerating visualizations...\n")
        
        filepaths = []
        
        # Order matters - most important first
        visualizations = [
            ("Executive Summary", self.plot_executive_summary),
            ("Engine Comparison Dashboard", self.plot_engine_comparison_dashboard),
            ("TTFT Detailed Analysis", self.plot_ttft_detailed_comparison),
            ("Throughput Analysis", self.plot_throughput_analysis),
            ("Inter-Token Latency Analysis", self.plot_inter_token_latency_analysis),
            ("Scalability Analysis", self.plot_scalability_analysis),
            ("Performance Heatmap", self.plot_performance_heatmap),
            ("Efficiency Analysis", self.plot_cost_efficiency_analysis),
            ("Reliability Metrics", self.plot_reliability_metrics),
        ]
        
        for i, (name, func) in enumerate(visualizations, 1):
            print(f"  [{i}/{len(visualizations)}] Creating {name}...", end=' ')
            try:
                filepath = func(save=True)
                if filepath:
                    filepaths.append(filepath)
                    print("✓")
                else:
                    print("⊘ (skipped - insufficient data)")
            except Exception as e:
                print(f"✗ Error: {str(e)}")
        
        print(f"\n{'='*80}")
        print(f"✓ Successfully generated {len(filepaths)} visualizations")
        print(f"  Output directory: {self.output_dir}/")
        print(f"{'='*80}\n")
        
        # Print recommendation
        print("📊 PRESENTATION TIPS:")
        print("  1. Start with '00_executive_summary' for stakeholder overview")
        print("  2. Use '01_engine_comparison_dashboard' for detailed metrics")
        print("  3. Show '04_inter_token_latency' for streaming applications")
        print("  4. Use '06_performance_heatmap' for at-a-glance comparison")
        print("  5. End with '08_reliability_metrics' for production readiness\n")
        
        return filepaths
    
    def generate_quick_comparison(self, save: bool = True) -> str:
        """
        Generate a single quick comparison chart - useful for rapid analysis.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Quick Performance Comparison: NIM vs vLLM', 
                     fontsize=18, fontweight='bold')
        
        grouped = self.df_success.groupby('inference_engine').agg({
            'time_to_first_token': 'mean',
            'tokens_per_second': 'mean',
            'total_latency': 'mean'
        }).reset_index()
        
        engines = grouped['inference_engine']
        colors = [self._get_engine_color(eng) for eng in engines]
        
        # TTFT
        axes[0].bar(engines, grouped['time_to_first_token'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[0].set_title('Time to First Token\n(Lower is Better)', fontweight='bold', pad=10)
        axes[0].set_ylabel('Seconds', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        self._add_value_labels(axes[0], '%.3f')
        
        # Throughput
        axes[1].bar(engines, grouped['tokens_per_second'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[1].set_title('Throughput\n(Higher is Better)', fontweight='bold', pad=10)
        axes[1].set_ylabel('Tokens/Second', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        self._add_value_labels(axes[1], '%.1f')
        
        # Total Latency
        axes[2].bar(engines, grouped['total_latency'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[2].set_title('Total Latency\n(Lower is Better)', fontweight='bold', pad=10)
        axes[2].set_ylabel('Seconds', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        self._add_value_labels(axes[2], '%.3f')
        
        plt.tight_layout()
        
        if save:
            return self._save_figure('quick_comparison')
        return None
