#!/bin/bash
# Run QWEN3-32B on policy violation tasks with teacher thinking

echo "=== QWEN3-32B Policy Violation Evaluation with Teacher Thinking ==="
echo ""
echo "This script demonstrates the concrete implementation where:"
echo "1. Teacher thinking is compressed from 9000+ to 800 chars"
echo "2. Only actionable SOQL queries are extracted"
echo "3. QWEN3 gets concrete execution guidance"
echo "4. Expected improvement: 0% → 20-25% accuracy"
echo ""

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/CRMArena"
export CUDA_VISIBLE_DEVICES=0

# Check if OpenAI key is set (needed for CRMArena evaluation)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. Loading from .env..."
    source CRMArena/.env
fi

# First, test the extraction logic
echo "Step 1: Testing teacher thinking extraction..."
python3 test_policy_violation_extraction.py

echo ""
echo "Step 2: Running QWEN3 with teacher guidance..."
echo ""

# Run the main evaluation
python3 qwen3_policy_violation_with_teacher.py \
    --model_path "Qwen/Qwen3-32B" \
    --teacher_traces "teacher_crmarena_traces_parallel/20250731_181710/policy_violation_identification_traces_gpu2.jsonl" \
    --output_file "qwen3_policy_results.json" \
    --compare_baseline

echo ""
echo "Evaluation complete! Check qwen3_policy_results.json for detailed results."