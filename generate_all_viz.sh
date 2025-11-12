#!/bin/bash
# Fine-grained Benchmark Visualization Generator
# Creates individual analyses per model family and concurrency level

set -e  # Exit on error

echo "🚀 Fine-Grained Benchmark Visualization Suite"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create viz directory
mkdir -p viz

# =============================================================================
# LLAMA FAMILY: Individual concurrency analyses (vLLM + NIM + Ollama)
# =============================================================================
echo -e "${BLUE}🦙 LLAMA FAMILY: Per-Concurrency Analysis${NC}"
echo "-------------------------------------------"

echo -e "${GREEN}  1. Llama Sequential (Baseline)...${NC}"
python3 generate_visualizations.py \
  --csv results/llama.csv results/llama-ollama.csv \
  --output-dir viz/llama_sequential \
  --title "Llama Sequential"

echo -e "${GREEN}  2. Llama C5...${NC}"
python3 generate_visualizations.py \
  --csv results/llama-5.csv results/llama-ollama-5.csv \
  --output-dir viz/llama_c5 \
  --title "Llama C5"

echo -e "${GREEN}  3. Llama C10...${NC}"
python3 generate_visualizations.py \
  --csv results/llama-10.csv results/llama-ollama-10.csv \
  --output-dir viz/llama_c10 \
  --title "Llama C10"

echo -e "${GREEN}  4. Llama C25...${NC}"
python3 generate_visualizations.py \
  --csv results/llama-25.csv results/llama-ollama-25.csv \
  --output-dir viz/llama_c25 \
  --title "Llama C25"

echo -e "${GREEN}  5. Llama C50 (Stress)...${NC}"
python3 generate_visualizations.py \
  --csv results/llama-50.csv results/llama-ollama-50.csv \
  --output-dir viz/llama_c50 \
  --title "Llama C50"

echo ""

# =============================================================================
# QWEN FAMILY: Individual concurrency analyses (vLLM + NIM + Ollama)
# =============================================================================
echo -e "${BLUE}🧠 QWEN FAMILY: Per-Concurrency Analysis${NC}"
echo "------------------------------------------"

echo -e "${GREEN}  6. Qwen Sequential (Baseline)...${NC}"
python3 generate_visualizations.py \
  --csv results/qwen.csv results/qwen-ollama.csv \
  --output-dir viz/qwen_sequential \
  --title "Qwen Sequential"

echo -e "${GREEN}  7. Qwen C5...${NC}"
python3 generate_visualizations.py \
  --csv results/qwen-5.csv results/qwen-ollama-5.csv \
  --output-dir viz/qwen_c5 \
  --title "Qwen C5"

echo -e "${GREEN}  8. Qwen C10...${NC}"
python3 generate_visualizations.py \
  --csv results/qwen-10.csv results/qwen-ollama-10.csv \
  --output-dir viz/qwen_c10 \
  --title "Qwen C10"

echo -e "${GREEN}  9. Qwen C25...${NC}"
python3 generate_visualizations.py \
  --csv results/qwen-25.csv results/qwen-ollama-25.csv \
  --output-dir viz/qwen_c25 \
  --title "Qwen C25"

echo -e "${GREEN}  10. Qwen C50 (Stress)...${NC}"
python3 generate_visualizations.py \
  --csv results/qwen-50.csv results/qwen-ollama-50.csv \
  --output-dir viz/qwen_c50 \
  --title "Qwen C50"

echo ""

# =============================================================================
# GEMMA FAMILY: Individual concurrency analyses (vLLM + NIM + Ollama)
# =============================================================================
echo -e "${BLUE}💎 GEMMA FAMILY: Per-Concurrency Analysis${NC}"
echo "-------------------------------------------"

echo -e "${GREEN}  11. Gemma Sequential (Baseline)...${NC}"
python3 generate_visualizations.py \
  --csv results/gemma.csv results/gemma-ollama.csv \
  --output-dir viz/gemma_sequential \
  --title "Gemma Sequential"

echo -e "${GREEN}  12. Gemma C5...${NC}"
python3 generate_visualizations.py \
  --csv results/gemma-5.csv results/gemma-ollama-5.csv \
  --output-dir viz/gemma_c5 \
  --title "Gemma C5"

echo -e "${GREEN}  13. Gemma C10...${NC}"
python3 generate_visualizations.py \
  --csv results/gemma-10.csv results/gemma-ollama-10.csv \
  --output-dir viz/gemma_c10 \
  --title "Gemma C10"

echo -e "${GREEN}  14. Gemma C25...${NC}"
python3 generate_visualizations.py \
  --csv results/gemma-25.csv results/gemma-ollama-25.csv \
  --output-dir viz/gemma_c25 \
  --title "Gemma C25"

echo -e "${GREEN}  15. Gemma C50 (Stress)...${NC}"
python3 generate_visualizations.py \
  --csv results/gemma-50.csv results/gemma-ollama-50.csv \
  --output-dir viz/gemma_c50 \
  --title "Gemma C50"

echo ""

# =============================================================================
# vLLM FAMILY: Old vs New comparison per concurrency (SPECIAL HANDLING)
# =============================================================================
echo -e "${BLUE}🔧 vLLM FAMILY: Old vs New Per-Concurrency${NC}"
echo "--------------------------------------------"

echo -e "${GREEN}  16. vLLM Sequential (Old vs New)...${NC}"
python3 generate_visualizations.py \
  --csv results/vllm.csv \
  --output-dir viz/vllm_sequential \
  --title "vLLM Sequential - Old vs New"

echo -e "${GREEN}  17. vLLM C5 (Old vs New)...${NC}"
python3 generate_visualizations.py \
  --csv results/vllm-5.csv \
  --output-dir viz/vllm_c5 \
  --title "vLLM C5 - Old vs New"

echo -e "${GREEN}  18. vLLM C10 (Old vs New)...${NC}"
python3 generate_visualizations.py \
  --csv results/vllm-10.csv \
  --output-dir viz/vllm_c10 \
  --title "vLLM C10 - Old vs New"

echo -e "${GREEN}  19. vLLM C25 (Old vs New)...${NC}"
python3 generate_visualizations.py \
  --csv results/vllm-25.csv \
  --output-dir viz/vllm_c25 \
  --title "vLLM C25 - Old vs New"

echo -e "${GREEN}  20. vLLM C50 (Old vs New - Stress)...${NC}"
python3 generate_visualizations.py \
  --csv results/vllm-50.csv \
  --output-dir viz/vllm_c50 \
  --title "vLLM C50 - Old vs New"

echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${YELLOW}✅ COMPLETE! All visualizations generated${NC}"
echo "=========================================="
echo ""
echo "📁 Results structure (20 fine-grained analyses):"
echo ""
echo "   🦙 LLAMA (5 analyses):"
echo "      viz/llama_sequential/  viz/llama_c5/  viz/llama_c10/  viz/llama_c25/  viz/llama_c50/"
echo ""
echo "   🧠 QWEN (5 analyses):"
echo "      viz/qwen_sequential/   viz/qwen_c5/   viz/qwen_c10/   viz/qwen_c25/   viz/qwen_c50/"
echo ""
echo "   💎 GEMMA (5 analyses):"
echo "      viz/gemma_sequential/  viz/gemma_c5/  viz/gemma_c10/  viz/gemma_c25/  viz/gemma_c50/"
echo ""
echo "   🔧 vLLM (5 analyses - Old vs New):"
echo "      viz/vllm_sequential/   viz/vllm_c5/   viz/vllm_c10/   viz/vllm_c25/   viz/vllm_c50/"
echo ""
echo "💡 Total: 120 visualizations (20 analyses × 6 charts each)"
echo ""
echo "🎯 Each analysis contains 6 charts:"
echo "   01_engine_comparison_dashboard"
echo "   02_ttft_detailed_comparison"
echo "   03_throughput_analysis"
echo "   04_inter_token_latency_analysis"
echo "   05_scalability_analysis"
echo "   06_performance_heatmap"

