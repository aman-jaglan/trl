#!/usr/bin/env python
"""
Evaluate student model on CRMArena policy violation tasks with teacher thinking.
This script loads tasks from HuggingFace, uses teacher thinking traces, and evaluates using CRMArena's system.
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

# Add CRMArena to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'CRMArena'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from crm_sandbox.data.assets import TASKS_B2B, B2B_SCHEMA
from crm_sandbox.env.env import ChatEnv
from crm_sandbox.agents.chat_agent import ChatAgent
from crm_sandbox.agents.prompts import SCHEMA_STRING, REACT_INTERNAL_PROMPT, SYSTEM_METADATA
from crm_sandbox.agents.utils import parse_wrapped_response


class StudentWithTeacherThinking(ChatAgent):
    """
    Student agent that uses teacher thinking at inference time.
    Inherits from ChatAgent to use CRMArena's execution and evaluation system.
    
    Note: This uses CRMArena's text-based tool format (<execute>/<respond> tags),
    not Qwen's native tool calling API. The model generates these tags as part of
    its text output, which CRMArena then parses and executes.
    """
    
    def __init__(
        self,
        schema_obj,
        student_model_path: str,
        teacher_traces: Dict[int, str],
        max_turns: int = 20,
        eval_mode: str = "default",
        strategy: str = "react",
        agent_type: str = "internal",
        device_map: str = "auto"
    ):
        # Initialize parent ChatAgent with dummy model (we'll override generation)
        super().__init__(
            schema_obj=schema_obj,
            model="gpt-4",  # Dummy, we'll use our own model
            max_turns=max_turns,
            eval_mode=eval_mode,
            strategy=strategy,
            agent_type=agent_type,
            provider="openai"  # Dummy provider
        )
        
        # Load student model
        print(f"Loading student model from {student_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        self.student_model.eval()
        
        # Store teacher traces
        self.teacher_traces = teacher_traces
        self.current_task_id = None
        
    def reset(self, args):
        """Reset agent for new task with teacher thinking."""
        # Store task ID for teacher trace lookup
        self.current_task_id = args.get('task_id')
        
        # Call parent reset to set up base prompt
        super().reset(args)
        
        # Get teacher thinking for this task
        teacher_thinking = self.teacher_traces.get(self.current_task_id, "")
        
        if teacher_thinking:
            # Log teacher thinking details
            print(f"\n[DEBUG] Task {self.current_task_id}: Teacher thinking loaded")
            print(f"  - Thinking length: {len(teacher_thinking)} characters")
            print(f"  - First 200 chars: {teacher_thinking[:200]}...")
            print(f"  - Contains 'steps I need to take': {'steps I need to take' in teacher_thinking}")
            
            # Create augmented user prompt with teacher thinking
            original_query = self.messages[-1]["content"]
            
            augmented_query = f"""Task: {original_query}

TEACHER'S THINKING PROCESS (for reference):
{teacher_thinking}

INSTRUCTIONS:
- Review the teacher's thinking above to understand their systematic approach
- Identify useful patterns like query structure, data analysis methods, and reasoning steps
- Pay special attention to the teacher's decision criteria and conditional reasoning
- Execute your solution using the CRM system

DECISION GUIDANCE:
- A policy violation exists ONLY when the agent's ACTIONS violated a documented procedure
- Just having an issue (like scalability) is NOT a violation unless the agent mishandled it
- Examples of actual violations:
  * Case closed without resolution when policy requires resolution
  * Case not escalated when policy requires escalation for that issue type
  * Wrong workflow followed for specific issue type
  * Agent didn't follow the procedure outlined in a knowledge article
- If the case was handled normally, answer "None" even if related knowledge articles exist
- The knowledge article ID should be returned ONLY if it documents a procedure the agent failed to follow

IMPORTANT FORMAT REQUIREMENTS:
- Use <execute> tags for SOQL/SOSL queries: <execute>SELECT Id FROM Case WHERE ...</execute>
- Use <respond> tags for your final answer: <respond>None</respond> or <respond>KA123</respond>
- First think through the problem, then execute queries, finally respond with the answer

Example decision patterns:
- Case closed without resolution + KB article requires resolution → Violation → Answer: ka0Wt000000EoD3IAK
- Case not escalated + KB article requires escalation for this issue → Violation → Answer: ka0Wt000000EnwwIAC
- Case has scalability issue but handled properly → No violation → Answer: None
- Case closed normally following standard procedure → No violation → Answer: None
- Issue exists but agent followed all procedures → No violation → Answer: None

Begin your solution:"""
            
            # Update the last message with augmented query
            self.messages[-1]["content"] = augmented_query
        else:
            print(f"\n[WARNING] Task {self.current_task_id}: No teacher thinking found!")
    
    def generate_response(self) -> str:
        """Generate response using student model instead of API."""
        # Create prompt from messages
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)  # 32k context
        inputs = {k: v.to(self.student_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.student_model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,  # Deterministic generation
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response
    
    def act(self, env, index=None, temperature=0.0):
        """Override act method to use our own generation."""
        query, metadata = env.reset(task_index=index)
        
        # Reset with task_id included
        self.reset({
            "query": query,
            "metadata": metadata,
            "task_id": index
        })
        
        self.info["observation_sizes"] = []
        done = False
        reward = 0
        observation = "None"
        
        for turn in range(self.max_turns):
            # Add observation to messages if not first turn
            if turn > 0 and observation and observation != "None":
                self.messages.append({"role": "user", "content": observation})
            
            # Generate response using student model
            response = self.generate_response()
            self.messages.append({"role": "assistant", "content": response})
            
            # Log student's response to see reasoning
            if turn == 0:  # First turn - show how student uses teacher thinking
                print(f"\n[STUDENT REASONING - Task {self.current_task_id}]")
                print(f"First 500 chars of response: {response[:500]}...")
                if 'teacher' in response.lower() or 'thinking' in response.lower():
                    print("  → Student explicitly references teacher thinking")
                if 'SELECT' in response and 'Case' in response:
                    print("  → Student uses similar query patterns")
            
            # Parse action from response
            action = self.message_action_parser({"content": response}, self.model)
            
            if action is None:
                # No valid action found, break
                break
            
            # Step in environment
            observation, reward, done, info = env.step(action)
            self.info["observation_sizes"].append(len(str(observation)))
            
            if done:
                # Store parsed answer if available
                if "end_reason" in info and "parsed_answer" in info["end_reason"]:
                    self.info["parsed_answer"] = info["end_reason"]["parsed_answer"]
                break
        
        # Update info
        self.info["num_turns"] = turn + 1
        self.info["total_cost"] = 0  # Local model, no API cost
        
        return reward


def load_teacher_traces(traces_path: str) -> Dict[int, str]:
    """Load teacher thinking traces indexed by task_id."""
    traces = {}
    
    with open(traces_path, 'r') as f:
        # Handle both JSON array and JSONL formats
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            data_list = json.loads(content)
        else:
            # JSONL format
            data_list = [json.loads(line) for line in content.split('\n') if line.strip()]
        
        for data in data_list:
            task_id = data['task_id']
            
            # Extract thinking from thinking_traces
            thinking_content = ""
            if 'thinking_traces' in data:
                for trace in data['thinking_traces']:
                    if trace.get('type') == 'think_tags' and trace.get('content'):
                        thinking_content = trace['content']
                        break
            
            # Clean up thinking to remove solution-specific content
            if thinking_content:
                lines = thinking_content.split('\n')
                cleaned_lines = []
                skip_keywords = [
                    'the answer is', 'solution:', 'final answer:', 
                    '<|begin_of_solution|>', '<|end_of_solution|>'
                ]
                
                for line in lines:
                    if not any(keyword in line.lower() for keyword in skip_keywords):
                        cleaned_lines.append(line)
                
                thinking_content = '\n'.join(cleaned_lines)
            
            traces[task_id] = thinking_content
    
    return traces


def evaluate_with_teacher_thinking(
    student_model_path: str,
    teacher_traces_path: str,
    output_file: str,
    max_tasks: Optional[int] = None,
    compare_baseline: bool = False,
    debug: bool = False
):
    """
    Evaluate student model on policy violation tasks with teacher thinking.
    """
    print(f"Loading student model: {student_model_path}")
    print(f"Loading teacher traces: {teacher_traces_path}")
    
    # Load teacher traces
    teacher_traces = load_teacher_traces(teacher_traces_path)
    print(f"Loaded {len(teacher_traces)} teacher thinking traces")
    
    # Load policy violation tasks from HuggingFace
    policy_tasks = [
        t for t in TASKS_B2B 
        if t.get("task") == "policy_violation_identification"
    ]
    
    # Debug: Show which task IDs have traces
    trace_task_ids = list(teacher_traces.keys())[:5]
    print(f"First 5 task IDs with traces: {trace_task_ids}")
    
    # Debug: Check if our policy task IDs match
    policy_task_ids = [t['idx'] for t in policy_tasks[:5]]
    print(f"First 5 policy task IDs: {policy_task_ids}")
    
    matching = set(policy_task_ids) & set(teacher_traces.keys())
    print(f"Matching task IDs: {len(matching)} out of {len(policy_tasks)}")
    
    if max_tasks:
        policy_tasks = policy_tasks[:max_tasks]
    
    print(f"Evaluating {len(policy_tasks)} policy violation tasks")
    
    # Create task dictionary for environment
    tasks_dict = {t['idx']: t for t in policy_tasks}
    
    # Initialize environment
    env = ChatEnv(
        tasks=tasks_dict,
        org_type="b2b",
        user_model="gpt-4",  # For evaluation/parsing
        user_provider="openai"
    )
    
    # Initialize student agent with teacher thinking
    agent_with_teacher = StudentWithTeacherThinking(
        schema_obj=B2B_SCHEMA,
        student_model_path=student_model_path,
        teacher_traces=teacher_traces,
        eval_mode="aided"  # Include optional metadata
    )
    
    # Run evaluation with teacher
    results_with_teacher = []
    
    # Debug: Show first prompt if requested
    if debug and policy_tasks:
        print("\n=== EXAMPLE PROMPT FOR FIRST TASK ===")
        first_task = policy_tasks[0]
        first_task_id = first_task['idx']
        
        # Create a debug agent to show the prompt
        debug_agent = StudentWithTeacherThinking(
            schema_obj=B2B_SCHEMA,
            student_model_path=student_model_path,
            teacher_traces=teacher_traces,
            eval_mode="aided"
        )
        
        # Reset with first task to see the prompt
        query = first_task['query']
        metadata = first_task['metadata']
        debug_agent.reset({
            "query": query,
            "metadata": metadata,
            "task_id": first_task_id
        })
        
        # Show the full prompt
        print("System Prompt:")
        print(debug_agent.messages[0]['content'][:500] + "...\n")
        print("User Prompt (with teacher thinking):")
        print(debug_agent.messages[1]['content'][:2000] + "...\n")
        print("=" * 50 + "\n")
    
    for task in tqdm(policy_tasks, desc="Evaluating with teacher thinking"):
        task_id = task['idx']
        
        try:
            # Run agent on task
            reward = agent_with_teacher.act(env, index=task_id)
            
            # Get parsed answer from agent's stored info
            parsed_answer = agent_with_teacher.info.get("parsed_answer", [])
            
            result = {
                "task_id": task_id,
                "ground_truth": task.get("answer"),
                "predicted": parsed_answer,
                "reward": reward,
                "success": reward == 1,
                "num_turns": agent_with_teacher.info.get("num_turns", 0),
                "with_teacher": True
            }
            
            # Optionally save full conversation for analysis
            if debug:
                result["conversation_preview"] = {
                    "first_response": agent_with_teacher.messages[2]["content"][:500] if len(agent_with_teacher.messages) > 2 else "",
                    "final_response": agent_with_teacher.messages[-1]["content"][:500] if agent_with_teacher.messages else ""
                }
            
        except Exception as e:
            print(f"Error on task {task_id}: {e}")
            result = {
                "task_id": task_id,
                "error": str(e),
                "success": False,
                "with_teacher": True
            }
        
        results_with_teacher.append(result)
    
    # Calculate accuracy
    successful = sum(1 for r in results_with_teacher if r.get('success', False))
    total = len(results_with_teacher)
    accuracy = (successful / total * 100) if total > 0 else 0
    
    print(f"\nResults with Teacher Thinking:")
    print(f"Total tasks: {total}")
    print(f"Successful: {successful}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Optionally run baseline without teacher
    results_baseline = []
    if compare_baseline:
        print("\nRunning baseline without teacher thinking...")
        
        # Create agent without teacher traces
        agent_baseline = StudentWithTeacherThinking(
            schema_obj=B2B_SCHEMA,
            student_model_path=student_model_path,
            teacher_traces={},  # Empty traces
            eval_mode="aided"
        )
        
        for task in tqdm(policy_tasks, desc="Evaluating baseline"):
            task_id = task['idx']
            
            try:
                reward = agent_baseline.act(env, index=task_id)
                parsed_answer = agent_baseline.info.get("parsed_answer", [])
                
                result = {
                    "task_id": task_id,
                    "ground_truth": task.get("answer"),
                    "predicted": parsed_answer,
                    "reward": reward,
                    "success": reward == 1,
                    "num_turns": agent_baseline.info.get("num_turns", 0),
                    "with_teacher": False
                }
                
            except Exception as e:
                print(f"Error on task {task_id}: {e}")
                result = {
                    "task_id": task_id,
                    "error": str(e),
                    "success": False,
                    "with_teacher": False
                }
            
            results_baseline.append(result)
        
        # Calculate baseline accuracy
        baseline_successful = sum(1 for r in results_baseline if r.get('success', False))
        baseline_accuracy = (baseline_successful / total * 100) if total > 0 else 0
        
        print(f"\nBaseline Results (without teacher):")
        print(f"Total tasks: {total}")
        print(f"Successful: {baseline_successful}")
        print(f"Accuracy: {baseline_accuracy:.1f}%")
        print(f"\nImprovement: {accuracy - baseline_accuracy:.1f}%")
    
    # Save results
    output_data = {
        "metadata": {
            "student_model": student_model_path,
            "teacher_traces": teacher_traces_path,
            "num_tasks": total,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "with_teacher": {
                "total": total,
                "successful": successful,
                "accuracy": accuracy
            }
        },
        "results_with_teacher": results_with_teacher
    }
    
    if compare_baseline:
        output_data["summary"]["baseline"] = {
            "total": total,
            "successful": baseline_successful,
            "accuracy": baseline_accuracy
        }
        output_data["summary"]["improvement"] = accuracy - baseline_accuracy
        output_data["results_baseline"] = results_baseline
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate student model with teacher thinking on CRMArena")
    
    parser.add_argument(
        "--student_model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Path or HuggingFace ID of student model"
    )
    
    parser.add_argument(
        "--teacher_traces",
        type=str,
        default="teacher_crmarena_traces_parallel/20250731_181710/policy_violation_identification_traces.json",
        help="Path to teacher thinking traces JSON file"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="student_evaluation_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate (default: all 100)"
    )
    
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Also run baseline evaluation without teacher thinking"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show example prompts and debug information"
    )
    
    args = parser.parse_args()
    
    evaluate_with_teacher_thinking(
        student_model_path=args.student_model,
        teacher_traces_path=args.teacher_traces,
        output_file=args.output_file,
        max_tasks=args.max_tasks,
        compare_baseline=args.compare_baseline,
        debug=args.debug
    )