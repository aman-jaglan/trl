#!/bin/bash
# Generate thinking traces using 4 GPUs in parallel

echo "Setting up environment..."

# Copy .env file if needed
if [ ! -f ".env" ]; then
    cp CRMArena/.env .env
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. The evaluation will need GPT access for exact match parsing."
    echo "Please set: export OPENAI_API_KEY=your_key"
fi

# Add CRMArena to Python path
export PYTHONPATH=/home/user/trl/CRMArena:$PYTHONPATH

echo ""
echo "Starting parallel trace generation on 4 GPUs..."
echo "This will distribute policy tasks across GPUs for faster processing"
echo ""

# Run parallel trace generation
python generate_crmarena_traces_parallel.py \
    --checkpoint_path /home/user/trl/RLT/results/rlt_teacher/2025.07.30192812/checkpoint-25 \
    --output_dir /home/user/trl/teacher_crmarena_traces_parallel \
    --org_type b2b \
    --num_gpus 4

echo ""
echo "Trace generation complete!"
echo "Check the output directory for results"