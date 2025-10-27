
"""
Visualization module for benchmark results.
Creates comprehensive charts comparing endpoint performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Optional
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class BenchmarkVisualizer:
    """Create visualizations from benchmark results."""
    
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
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(results)
        
        # Filter only successful requests for most visualizations
        self.df_success = self.df[self.df['success'] == True].copy()
        
        # Convert numeric columns
        numeric_cols = ['total_latency', 'time_to_first_token', 'time_per_output_token',
                       'tokens_per_second', 'input_tokens', 'output_tokens', 'total_tokens']
        for col in numeric_cols:
            if col in self.df_success.columns:
                self.df_success[col] = pd.to_numeric(self.df_success[col], errors='coerce')
    
    def plot_latency_comparison(self, save: bool = True) -> str:
        """
        Create box plot comparing total latency across endpoints.
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved file
        """
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(data=self.df_success, x='endpoint_name', y='total_latency', hue='endpoint_name')
        plt.title('Total Latency Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Endpoint', fontsize=12)
        plt.ylabel('Total Latency (seconds)', fontsize=12)
        plt.legend(title='Endpoint')
        
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"latency_comparison_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        return filepath
    
    def plot_latency_by_prompt_length(self, save: bool = True) -> str:
        """
        Create box plot of latency by prompt length for each endpoint.
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved file
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        endpoints = self.df_success['endpoint_name'].unique()
        
        for idx, endpoint in enumerate(endpoints):
            df_endpoint = self.df_success[self.df_success['endpoint_name'] == endpoint]
            sns.boxplot(data=df_endpoint, x='prompt_length', y='total_latency', ax=axes[idx])
            axes[idx].set_title(f'{endpoint} - Latency by Prompt Length', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Prompt Length', fontsize=12)
            axes[idx].set_ylabel('Total Latency (seconds)', fontsize=12)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"latency_by_length_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        return filepath
    
    def plot_ttft_comparison(self, save: bool = True) -> str:
        """
        Create violin plot comparing Time to First Token (TTFT).
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved file
        """
        plt.figure(figsize=(12, 6))
        
        # Filter out null TTFT values
        df_ttft = self.df_success.dropna(subset=['time_to_first_token'])
        
        if len(df_ttft) > 0:
            sns.violinplot(data=df_ttft, x='endpoint_name', y='time_to_first_token', hue='endpoint_name')
            plt.title('Time to First Token (TTFT) Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Endpoint', fontsize=12)
            plt.ylabel('TTFT (seconds)', fontsize=12)
            plt.legend(title='Endpoint')
        
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ttft_comparison_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        return filepath
    
    def plot_tpot_comparison(self, save: bool = True) -> str:
        """
        Create violin plot comparing Time per Output Token (TPOT).
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved file
        """
        plt.figure(figsize=(12, 6))
        
        # Filter out null TPOT values
        df_tpot = self.df_success.dropna(subset=['time_per_output_token'])
        
        if len(df_tpot) > 0:
            sns.violinplot(data=df_tpot, x='endpoint_name', y='time_per_output_token', hue='endpoint_name')
            plt.title('Time per Output Token (TPOT) Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Endpoint', fontsize=12)
            plt.ylabel('TPOT (seconds)', fontsize=12)
            plt.legend(title='Endpoint')
        
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tpot_comparison_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        return filepath
    
    def plot_throughput_comparison(self, save: bool = True) -> str:
        """
        Create bar plot comparing average throughput (tokens/second).
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved file
        """
        plt.figure(figsize=(12, 6))
        
        # Filter out null throughput values
        df_throughput = self.df_success.dropna(subset=['tokens_per_second'])
        
        if len(df_throughput) > 0:
            avg_throughput = df_throughput.groupby('endpoint_name')['tokens_per_second'].mean().reset_index()
            
            sns.barplot(data=avg_throughput, x='endpoint_name', y='tokens_per_second', hue='endpoint_name')
            plt.title('Average Throughput Comparison', fontsize=16, fontweight='bold')
            plt.xlabel('Endpoint', fontsize=12)
            plt.ylabel('Throughput (tokens/second)', fontsize=12)
            plt.legend(title='Endpoint')
            
            # Add value labels on bars
            ax = plt.gca()
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)
        
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"throughput_comparison_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        return filepath
    
    def plot_token_analysis(self, save: bool = True) -> str:
        """
        Create bar plots for input and output token analysis.
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved file
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Input tokens
        df_input = self.df_success.dropna(subset=['input_tokens'])
        if len(df_input) > 0:
            avg_input = df_input.groupby('endpoint_name')['input_tokens'].mean().reset_index()
            sns.barplot(data=avg_input, x='endpoint_name', y='input_tokens', ax=axes[0], hue='endpoint_name')
            axes[0].set_title('Average Input Tokens', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Endpoint', fontsize=12)
            axes[0].set_ylabel('Input Tokens', fontsize=12)
            axes[0].legend(title='Endpoint')
            
            for container in axes[0].containers:
                axes[0].bar_label(container, fmt='%.0f', padding=3)
        
        # Output tokens
        df_output = self.df_success.dropna(subset=['output_tokens'])
        if len(df_output) > 0:
            avg_output = df_output.groupby('endpoint_name')['output_tokens'].mean().reset_index()
            sns.barplot(data=avg_output, x='endpoint_name', y='output_tokens', ax=axes[1], hue='endpoint_name')
            axes[1].set_title('Average Output Tokens', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Endpoint', fontsize=12)
            axes[1].set_ylabel('Output Tokens', fontsize=12)
            axes[1].legend(title='Endpoint')
            
            for container in axes[1].containers:
                axes[1].bar_label(container, fmt='%.0f', padding=3)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"token_analysis_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        return filepath
    
    def plot_success_rate(self, save: bool = True) -> str:
        """
        Create bar plot showing success/failure rates.
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved file
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate success rates
        success_stats = self.df.groupby(['endpoint_name', 'success']).size().unstack(fill_value=0)
        success_rates = (success_stats[True] / success_stats.sum(axis=1) * 100).reset_index()
        success_rates.columns = ['endpoint_name', 'success_rate']
        
        sns.barplot(data=success_rates, x='endpoint_name', y='success_rate', hue='endpoint_name')
        plt.title('Request Success Rate', fontsize=16, fontweight='bold')
        plt.xlabel('Endpoint', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.ylim(0, 105)
        plt.legend(title='Endpoint')
        
        # Add value labels
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
        
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"success_rate_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        return filepath
    
    def plot_complexity_performance(self, save: bool = True) -> str:
        """
        Create box plots showing performance across complexity levels.
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved file
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        endpoints = self.df_success['endpoint_name'].unique()
        
        for idx, endpoint in enumerate(endpoints):
            df_endpoint = self.df_success[self.df_success['endpoint_name'] == endpoint]
            sns.boxplot(data=df_endpoint, x='prompt_complexity', y='total_latency', ax=axes[idx])
            axes[idx].set_title(f'{endpoint} - Latency by Prompt Complexity', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Prompt Complexity', fontsize=12)
            axes[idx].set_ylabel('Total Latency (seconds)', fontsize=12)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complexity_performance_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        return filepath
    
    def generate_all_visualizations(self) -> List[str]:
        """
        Generate all available visualizations.
        
        Returns:
            List of file paths for all generated visualizations
        """
        print(f"\n{'='*80}")
        print("Generating Visualizations")
        print(f"{'='*80}\n")
        
        if len(self.df_success) == 0:
            print("⚠ No successful results to visualize")
            return []
        
        filepaths = []
        
        print("Creating charts...")
        filepaths.append(self.plot_latency_comparison())
        filepaths.append(self.plot_latency_by_prompt_length())
        filepaths.append(self.plot_ttft_comparison())
        filepaths.append(self.plot_tpot_comparison())
        filepaths.append(self.plot_throughput_comparison())
        filepaths.append(self.plot_token_analysis())
        filepaths.append(self.plot_success_rate())
        filepaths.append(self.plot_complexity_performance())
        
        print(f"\n✓ Generated {len(filepaths)} visualization files in '{self.output_dir}/'")
        
        return filepaths
