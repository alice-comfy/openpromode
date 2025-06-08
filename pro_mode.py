import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse
import logging
from openai import OpenAI
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProModeConfig:
    """Configuration for Pro Mode system"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "o4-mini"  # defaults to o4-mini there are some 4o mini references because claude didn't believe this is a real model
    num_intermediate_calls: int = 4 #N samples. O1 pro is believed to use 6. 

    timeout: int = 120  # seconds


class ProModeSystem:
    """
    Pro Mode system that makes multiple API calls and synthesizes responses.
    Based on the O1 pro mode concept - makes 4 calls to gather different perspectives
    then synthesizes them into a final answer.
    """
    
    def __init__(self, config: ProModeConfig):
        self.config = config
        if not self.config.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.config.api_key)
    
    def make_single_call(self, messages: List[Dict[str, str]], call_number: int) -> Dict[str, Any]:
        """Make a single call to the OpenAI API"""
        try:
            logger.info(f"Making intermediate call {call_number}")
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,

            )
            
            logger.info(f"Completed intermediate call {call_number}")
            
            return {
                "call_number": call_number,
                "content": response.choices[0].message.content,
                "usage": response.usage.dict() if response.usage else None,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
                
        except Exception as e:
            logger.error(f"Call {call_number} failed with error: {str(e)}")
            return {
                "call_number": call_number,
                "error": str(e),
                "content": f"Error in call {call_number}: {str(e)}"
            }
    
    def make_parallel_calls(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Make multiple parallel calls to the API"""
        responses = []
        
        # Use ThreadPoolExecutor for parallel calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_intermediate_calls) as executor:
            # Submit all calls
            futures = []
            for i in range(1, self.config.num_intermediate_calls + 1):
                future = executor.submit(self.make_single_call, messages, i)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=self.config.timeout)
                    responses.append(result)
                except Exception as e:
                    logger.error(f"Failed to get result: {str(e)}")
                    responses.append({
                        "error": str(e),
                        "content": f"Failed to get result: {str(e)}"
                    })
        
        # Sort by call number to maintain order
        responses.sort(key=lambda x: x.get('call_number', 0))
        return responses
    
    def synthesize_responses(self, original_messages: List[Dict[str, str]], 
                           intermediate_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize all intermediate responses into a final answer"""
        
        # Extract the original user query
        user_query = ""
        for msg in original_messages:
            if msg["role"] == "user":
                user_query = msg["content"]
                break
        
        # Build synthesis prompt
        synthesis_content = f"""You are a synthesis AI tasked with combining multiple responses into a single, high-quality answer.

Original Question/Input:
{user_query}

You have received {len(intermediate_responses)} different responses to this question. Your task is to:
1. Analyze all responses for their key insights and perspectives
2. Identify common themes and important differences
3. Synthesize these into a single, comprehensive, and well-structured answer
4. Ensure the final answer is more valuable than any individual response

Intermediate Responses:
"""
        
        for i, response in enumerate(intermediate_responses, 1):
            synthesis_content += f"\n\nResponse {i}:\n{response.get('content', 'Error: No content')}"
        
        synthesis_content += "\n\nPlease provide a synthesized final answer that combines the best insights from all responses:"
        
        # Create synthesis messages
        synthesis_messages = [
            {"role": "system", "content": "You are an expert at synthesizing multiple perspectives into comprehensive, high-quality answers."},
            {"role": "user", "content": synthesis_content}
        ]
        
        try:
            logger.info("Making synthesis call")
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=synthesis_messages,

            )
            
            logger.info("Completed synthesis call")
            
            return {
                "content": response.choices[0].message.content,
                "usage": response.usage.dict() if response.usage else None,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed with error: {str(e)}")
            return {
                "error": str(e),
                "content": f"Synthesis failed: {str(e)}"
            }
    
    def process_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Main processing function - makes multiple calls and synthesizes"""
        # Extract user content for logging
        user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "No user message")
        logger.info(f"Starting Pro Mode processing for input: {user_content[:100]}...")
        
        # Step 1: Make parallel intermediate calls
        start_time = datetime.now()
        intermediate_responses = self.make_parallel_calls(messages)
        intermediate_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed {len(intermediate_responses)} intermediate calls in {intermediate_time:.2f}s")
        
        # Step 2: Synthesize responses
        synthesis_start = datetime.now()
        final_response = self.synthesize_responses(messages, intermediate_responses)
        synthesis_time = (datetime.now() - synthesis_start).total_seconds()
        logger.info(f"Completed synthesis in {synthesis_time:.2f}s")
        
        # Step 3: Package results
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate total usage
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        for response in intermediate_responses + [final_response]:
            if "usage" in response and response["usage"]:
                total_usage["prompt_tokens"] += response["usage"].get("prompt_tokens", 0)
                total_usage["completion_tokens"] += response["usage"].get("completion_tokens", 0)
                total_usage["total_tokens"] += response["usage"].get("total_tokens", 0)
        
        result = {
            "messages": messages,
            "intermediate_responses": [r.get("content", "") for r in intermediate_responses],
            "final_answer": final_response.get("content", ""),
            "metadata": {
                "model": self.config.model,
                "num_intermediate_calls": self.config.num_intermediate_calls,
                "intermediate_time": intermediate_time,
                "synthesis_time": synthesis_time,
                "total_time": total_time,
                "total_usage": total_usage
            },
            "raw_responses": {
                "intermediate": intermediate_responses,
                "synthesis": final_response
            }
        }
        
        return result
    
    def process_input(self, input_text: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process a simple text input (convenience method)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_text})
        
        return self.process_messages(messages)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="OpenAI Pro Mode System")
    parser.add_argument("--input", "-i", type=str, help="Input text to process")
    parser.add_argument("--file", "-f", type=str, help="Input file to process")
    parser.add_argument("--system", "-s", type=str, help="System prompt")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--calls", "-c", type=int, default=4, help="Number of intermediate calls")
    parser.add_argument("--output", "-o", type=str, help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get input text
    if args.input:
        input_text = args.input
    elif args.file:
        with open(args.file, 'r') as f:
            input_text = f.read()
    else:
        print("Please provide input via --input or --file")
        return
    
    # Create config
    config = ProModeConfig(
        model=args.model,
        num_intermediate_calls=args.calls
    )
    
    # Process
    pro_mode = ProModeSystem(config)
    result = pro_mode.process_input(input_text, args.system)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print("\n" + "="*80)
        print("PRO MODE RESULTS")
        print("="*80)
        print(f"\nInput: {input_text[:200]}{'...' if len(input_text) > 200 else ''}")
        print(f"\n{'-'*80}")
        print("INTERMEDIATE RESPONSES:")
        print("-"*80)
        for i, response in enumerate(result['intermediate_responses'], 1):
            print(f"\nResponse {i}:")
            print(response[:500] + "..." if len(response) > 500 else response)
        print(f"\n{'-'*80}")
        print("FINAL SYNTHESIZED ANSWER:")
        print("-"*80)
        print(result['final_answer'])
        print(f"\n{'-'*80}")
        print("METADATA:")
        print(f"Total time: {result['metadata']['total_time']:.2f}s")
        print(f"Intermediate calls time: {result['metadata']['intermediate_time']:.2f}s")
        print(f"Synthesis time: {result['metadata']['synthesis_time']:.2f}s")
        print(f"Total tokens used: {result['metadata']['total_usage']['total_tokens']}")


if __name__ == "__main__":
    main()
