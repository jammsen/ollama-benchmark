#!/usr/bin/env python3
"""
LLM Benchmark tool for Ollama

A lightweight tool for measuring LLM performance metrics via Ollama:
- Token processing speed (t/s)
- Model load time
- Prompt evaluation time
- Response generation time

Usage:
    python benchmark.py [-v] [-m MODEL_NAMES...] [-p PROMPTS...] [-t] 

Example:
    python benchmark.py --verbose --models llama3.2:1b qwen3:4b "hello there" --table
"""

import argparse
from typing import List, Dict, Optional
from datetime import datetime
import os
import time

import ollama
from pydantic import BaseModel, Field

from tabulate import tabulate
from rich.console import Console, RenderableType
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group

# Ollama client will be initialized after parsing arguments
ollama_client = None


class Message(BaseModel):
    """Represents a single message in the chat interaction."""
    role: str
    content: str


class OllamaResponse(BaseModel):
    """
    Represents a structured response from the Ollama API.
    Contains performance metrics and message content.
    """
    model: str
    created_at: datetime | None = None
    message: Message
    done: bool
    total_duration: int = Field(default=0)
    load_duration: int = Field(default=0)
    prompt_eval_count: int = Field(default=0)
    prompt_eval_duration: int = Field(default=0)
    eval_count: int = Field(default=0)
    eval_duration: int = Field(default=0)

    @classmethod
    def from_chat_response(cls, response) -> 'OllamaResponse':
        """
        Converts an Ollama API response into an OllamaResponse instance.
        
        Args:
            response: Raw response from Ollama API
        
        Returns:
            OllamaResponse: Structured response object
        """
        return cls(
            model=response.model,
            message=Message(
                role=response.message.role,
                content=response.message.content
            ),
            done=response.done,
            total_duration=getattr(response, 'total_duration', 0),
            load_duration=getattr(response, 'load_duration', 0),
            prompt_eval_count=getattr(response, 'prompt_eval_count', 0),
            prompt_eval_duration=getattr(response, 'prompt_eval_duration', 0),
            eval_count=getattr(response, 'eval_count', 0),
            eval_duration=getattr(response, 'eval_duration', 0)
        )


def run_benchmark(
        model_name: str,
        prompt: str,
        verbose: bool,
        num_gpu: Optional[int] = None
) -> Optional[OllamaResponse]:
    """
    Executes a benchmark run for a specific model and prompt.

    Args:
        model_name: Name of the Ollama model to benchmark
        prompt: Input text to send to the model
        verbose: If True, prints streaming output
        num_gpu: Number of layers to offload to GPU (None = auto/default)

    Returns:
        OllamaResponse object containing benchmark results, or None if failed
    """
    messages = [{"role": "user", "content": prompt}]
    
    # Build options dict for GPU offloading if specified
    options = {}
    if num_gpu is not None:
        options['num_gpu'] = num_gpu

    try:
        if verbose:
            # For verbose mode, we'll collect the content while streaming
            content = ""
            stream = ollama_client.chat(
                model=model_name,
                messages=messages,
                stream=True,
                options=options if options else None,
            )
            for chunk in stream:
                if hasattr(chunk.message, 'content'):
                    content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)

            if not content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response. Please check if:")
                print("1. The model is properly loaded")
                print("2. The Ollama server is functioning correctly")
                print("3. Try running 'ollama run {model_name}' in terminal to verify model output")
                return None

            # Make a non-streaming call to get the metrics
            response = ollama_client.chat(
                model=model_name,
                messages=messages,
                options=options if options else None,
            )

            # Check if response has content
            if not hasattr(response.message, 'content') or not response.message.content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response in non-streaming mode")
                return None

            # Create response with collected content and metrics
            return OllamaResponse(
                model=model_name,
                message=Message(
                    role="assistant",
                    content=content
                ),
                done=True,
                total_duration=getattr(response, 'total_duration', 0),
                load_duration=getattr(response, 'load_duration', 0),
                prompt_eval_count=getattr(response, 'prompt_eval_count', 0),
                prompt_eval_duration=getattr(response, 'prompt_eval_duration', 0),
                eval_count=getattr(response, 'eval_count', 0),
                eval_duration=getattr(response, 'eval_duration', 0)
            )
        else:
            # For non-verbose mode, just make a single non-streaming call
            response = ollama_client.chat(
                model=model_name,
                messages=messages,
                options=options if options else None,
            )

            # Check if response has content
            if not hasattr(response.message, 'content') or not response.message.content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response. Please check if:")
                print("1. The model is properly loaded")
                print("2. The Ollama server is functioning correctly")
                print("3. Try running 'ollama run {model_name}' in terminal to verify model output")
                return None

            return OllamaResponse.from_chat_response(response)

    except Exception as e:
        print(f"Error benchmarking {model_name}: {str(e)}")
        return None


def nanosec_to_sec(nanosec: int) -> float:
    """Converts nanoseconds to seconds."""
    return nanosec / 1_000_000_000


def inference_stats(model_response: OllamaResponse) -> None:
    """
    Calculates and prints detailed inference statistics for a model response.

    Args:
        model_response: OllamaResponse containing benchmark metrics
    """
    # Calculate tokens per second for different phases
    prompt_ts = model_response.prompt_eval_count / (
        nanosec_to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (
        nanosec_to_sec(model_response.eval_duration)
    )
    total_ts = (
                       model_response.prompt_eval_count + model_response.eval_count
               ) / (
                   nanosec_to_sec(
                       model_response.prompt_eval_duration + model_response.eval_duration
                   )
               )

    print(
        f"""
----------------------------------------------------
        Model: {model_response.model}
        Performance Metrics:
            Prompt Processing:  {prompt_ts:.2f} tokens/sec
            Generation Speed:   {response_ts:.2f} tokens/sec
            Combined Speed:     {total_ts:.2f} tokens/sec

        Workload Stats:
            Input Tokens:       {model_response.prompt_eval_count}
            Generated Tokens:   {model_response.eval_count}
            Model Load Time:    {nanosec_to_sec(model_response.load_duration):.2f}s
            Processing Time:    {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
            Generation Time:    {nanosec_to_sec(model_response.eval_duration):.2f}s
            Total Time:         {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]) -> None:
    """
    Calculates and prints average statistics across multiple benchmark runs.

    Args:
        responses: List of OllamaResponse objects from multiple runs
    """
    if not responses:
        print("No stats to average")
        return

    # Calculate aggregate metrics
    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)


def average_stats_multi_run(model_name: str, all_runs: List[List[OllamaResponse]], runs: int) -> None:
    """
    Calculates and prints statistics for multiple runs, showing individual run stats and overall average.

    Args:
        model_name: Name of the model being benchmarked
        all_runs: List of Lists of OllamaResponse objects (outer list = runs, inner list = prompts)
        runs: Number of runs performed
    """
    if not all_runs:
        print("No stats to display")
        return

    print(f"\n{'='*60}")
    print(f"Results for model: {model_name}")
    print(f"{'='*60}")

    # Show individual run stats
    for run_idx, responses in enumerate(all_runs, 1):
        if not responses:
            continue
            
        if runs > 1:
            print(f"\n--- Run {run_idx}/{runs} ---")
        
        # Calculate aggregate metrics for this run
        res = OllamaResponse(
            model=model_name,
            created_at=datetime.now(),
            message=Message(
                role="system",
                content=f"Run {run_idx}",
            ),
            done=True,
            total_duration=sum(r.total_duration for r in responses),
            load_duration=sum(r.load_duration for r in responses),
            prompt_eval_count=sum(r.prompt_eval_count for r in responses),
            prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
            eval_count=sum(r.eval_count for r in responses),
            eval_duration=sum(r.eval_duration for r in responses),
        )
        inference_stats(res)

    # Show average across all runs
    if runs > 1:
        print(f"\n--- Average across {runs} runs ---")
        all_responses = [resp for run in all_runs for resp in run]
        num_runs = len(all_runs)
        
        if all_responses:
            avg_res = OllamaResponse(
                model=model_name,
                created_at=datetime.now(),
                message=Message(
                    role="system",
                    content=f"Average across {num_runs} runs",
                ),
                done=True,
                total_duration=int(sum(r.total_duration for r in all_responses) / num_runs),
                load_duration=int(sum(r.load_duration for r in all_responses) / num_runs),
                prompt_eval_count=int(sum(r.prompt_eval_count for r in all_responses) / num_runs),
                prompt_eval_duration=int(sum(r.prompt_eval_duration for r in all_responses) / num_runs),
                eval_count=int(sum(r.eval_count for r in all_responses) / num_runs),
                eval_duration=int(sum(r.eval_duration for r in all_responses) / num_runs),
            )
            inference_stats(avg_res)


def table_stats(benchmarks: Dict[str, List[List[OllamaResponse]]], runs: int) -> None:
    """
    Calculates and prints statistics across multiple benchmark runs and models, output as table.
    Shows individual runs and averages with Rich formatting.

    Args:
        benchmarks: Dict of modelNames and List of Lists of OllamaResponse objects (outer list = runs, inner list = prompts)
        runs: Number of runs performed
    """
    if not benchmarks:
        print("No results to output")
        return

    console = Console()
    console.print("\n[bold cyan]Table stats:[/bold cyan]")
    
    # Create Rich table
    table = Table(title="[bold magenta]Benchmark Results[/bold magenta]", show_header=True, header_style="bold cyan")
    
    table.add_column("Model\nName", style="cyan", no_wrap=True)
    table.add_column("Run", style="yellow")
    table.add_column("Prompt\nEvaluation Rate\n(T/s)", style="green", justify="right")
    table.add_column("Evaluation\nRate\n(T/s)", style="green", justify="right")
    table.add_column("Total\nRate\n(T/s)", style="green", justify="right")
    table.add_column("Load Time\n(s)", style="white", justify="right")
    table.add_column("Prompt\nEvaluation Count", style="white", justify="right")
    table.add_column("Prompt\nEvaluation Time\n(s)", style="white", justify="right")
    table.add_column("Evaluation\nCount", style="white", justify="right")
    table.add_column("Evaluation\nTime\n(s)", style="white", justify="right")
    table.add_column("Total Time\n(s)", style="white", justify="right")
    
    for model_name, all_runs in benchmarks.items():
        # Show each individual run
        for run_idx, responses in enumerate(all_runs, 1):
            if not responses:
                continue
                
            # Calculate aggregate metrics for this run
            total_duration = sum(r.total_duration for r in responses)
            load_duration = sum(r.load_duration for r in responses)
            prompt_eval_count = sum(r.prompt_eval_count for r in responses)
            prompt_eval_duration = sum(r.prompt_eval_duration for r in responses)
            eval_count = sum(r.eval_count for r in responses)
            eval_duration = sum(r.eval_duration for r in responses)

            # Calculate tokens per second for different phases
            prompt_ts = prompt_eval_count / nanosec_to_sec(prompt_eval_duration) if prompt_eval_duration > 0 else 0
            response_ts = eval_count / nanosec_to_sec(eval_duration) if eval_duration > 0 else 0
            total_ts = (prompt_eval_count + eval_count) / nanosec_to_sec(prompt_eval_duration + eval_duration) if (prompt_eval_duration + eval_duration) > 0 else 0

            run_label = f"Run {run_idx}" if runs > 1 else ""
            table.add_row(
                model_name,
                run_label,
                f"{prompt_ts:.2f}",
                f"{response_ts:.2f}",
                f"{total_ts:.2f}",
                f"{nanosec_to_sec(load_duration):.2f}",
                f"{prompt_eval_count:.2f}",
                f"{nanosec_to_sec(prompt_eval_duration):.2f}",
                f"{eval_count:.2f}",
                f"{nanosec_to_sec(eval_duration):.2f}",
                f"{nanosec_to_sec(total_duration):.2f}"
            )
        
        # Calculate and show average across all runs
        if runs > 1 and all_runs:
            # Flatten all responses from all runs
            all_responses = [resp for run in all_runs for resp in run]
            num_runs = len(all_runs)
            
            if all_responses:
                total_duration = sum(r.total_duration for r in all_responses) / num_runs
                load_duration = sum(r.load_duration for r in all_responses) / num_runs
                prompt_eval_count = sum(r.prompt_eval_count for r in all_responses) / num_runs
                prompt_eval_duration = sum(r.prompt_eval_duration for r in all_responses) / num_runs
                eval_count = sum(r.eval_count for r in all_responses) / num_runs
                eval_duration = sum(r.eval_duration for r in all_responses) / num_runs

                prompt_ts = prompt_eval_count / nanosec_to_sec(prompt_eval_duration) if prompt_eval_duration > 0 else 0
                response_ts = eval_count / nanosec_to_sec(eval_duration) if eval_duration > 0 else 0
                total_ts = (prompt_eval_count + eval_count) / nanosec_to_sec(prompt_eval_duration + eval_duration) if (prompt_eval_duration + eval_duration) > 0 else 0

                table.add_row(
                    f"[bold]{model_name}[/bold]",
                    "[bold yellow]Average[/bold yellow]",
                    f"[bold]{prompt_ts:.2f}[/bold]",
                    f"[bold]{response_ts:.2f}[/bold]",
                    f"[bold]{total_ts:.2f}[/bold]",
                    f"[bold]{nanosec_to_sec(load_duration):.2f}[/bold]",
                    f"[bold]{prompt_eval_count:.2f}[/bold]",
                    f"[bold]{nanosec_to_sec(prompt_eval_duration):.2f}[/bold]",
                    f"[bold]{eval_count:.2f}[/bold]",
                    f"[bold]{nanosec_to_sec(eval_duration):.2f}[/bold]",
                    f"[bold]{nanosec_to_sec(total_duration):.2f}[/bold]"
                )

    console.print(table)
    console.print()


def get_benchmark_models(test_models: List[str] = []) -> List[str]:
    """
    Retrieves and validates the list of models to benchmark.

    Args:
        test_models: List of specific models to test

    Returns:
        List of validated model names available for benchmarking
    """
    response = ollama_client.list()
    available_models = [model.get("model") for model in response.get("models", [])]

    if not test_models:
        # Use a default subset of models if none specified
        default_models = ["llama3", "mistral", "codellama", "deepseek", "gpt-oss", "gemma"]  # Common default models
        model_names = [m for m in available_models if any(d in m for d in default_models)]
        if not model_names:
            model_names = available_models[:3]  # Take first 3 available models if no defaults found
        # sort default subset alphabetically
        model_names.sort()
    else:
        # Filter requested models against available ones
        model_names = [model for model in test_models if model in available_models]
        if len(model_names) < len(test_models):
            missing_models = set(test_models) - set(available_models)
            print(f"Warning: Some requested models are not available: {missing_models}")
            
            # Try to pull missing models
            for model in missing_models:
                print(f"Attempting to pull model: {model}")
                try:
                    # Stream the pull progress
                    current_digest = None
                    for progress in ollama_client.pull(model, stream=True):
                        # Display progress information
                        if 'status' in progress:
                            status = progress['status']
                            
                            # Track digest to avoid repeating messages
                            digest = progress.get('digest', '')
                            if digest and digest != current_digest:
                                current_digest = digest
                                print(f"\\n{status}: {digest[:12]}...")
                            
                            # Show download progress
                            if 'completed' in progress and 'total' in progress:
                                completed = progress['completed']
                                total = progress['total']
                                percentage = (completed / total * 100) if total > 0 else 0
                                completed_mb = completed / (1024 * 1024)
                                total_mb = total / (1024 * 1024)
                                print(f"\\r{status}: {completed_mb:.1f}/{total_mb:.1f} MB ({percentage:.1f}%)", end="", flush=True)
                            elif status and not digest:
                                # For status messages without progress bars
                                print(f"\\r{status}", end="", flush=True)
                    
                    print(f"\\n✓ Successfully pulled {model}")
                    model_names.append(model)
                except Exception as e:
                    print(f"Failed to pull {model}: {str(e)}")

    if not model_names:
        raise RuntimeError("No valid models found for benchmarking")

    print(f"Evaluating models: {model_names}\n")
    return model_names


def create_stats_table(model_name: str, prompt_processing: float, generation_speed: float,
                      combined_speed: float, input_tokens: int, generated_tokens: int,
                      load_time: float, processing_time: float, generation_time: float,
                      total_time: float) -> Table:
    """
    Creates a Rich Table for displaying statistics.
    
    Args:
        model_name: Name of the model
        prompt_processing: Prompt processing speed in tokens/sec
        generation_speed: Generation speed in tokens/sec
        combined_speed: Combined speed in tokens/sec
        input_tokens: Number of input tokens
        generated_tokens: Number of generated tokens
        load_time: Model load time in seconds
        processing_time: Processing time in seconds
        generation_time: Generation time in seconds
        total_time: Total time in seconds
        
    Returns:
        Rich Table object with statistics
    """
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    table.add_row("[bold]Model[/bold]", model_name)
    table.add_row("", "")
    table.add_row("[bold]Performance Metrics[/bold]", "")
    table.add_row("  Prompt Processing", f"{prompt_processing:.2f} tokens/sec")
    table.add_row("  Generation Speed", f"{generation_speed:.2f} tokens/sec")
    table.add_row("  Combined Speed", f"{combined_speed:.2f} tokens/sec")
    table.add_row("", "")
    table.add_row("[bold]Workload Stats[/bold]", "")
    table.add_row("  Input Tokens", str(input_tokens))
    table.add_row("  Generated Tokens", str(generated_tokens))
    table.add_row("  Model Load Time", f"{load_time:.2f}s")
    table.add_row("  Processing Time", f"{processing_time:.2f}s")
    table.add_row("  Generation Time", f"{generation_time:.2f}s")
    table.add_row("  Total Time", f"{total_time:.2f}s")
    
    return table


def create_ollama_ps_table() -> Table:
    """
    Creates a Rich Table showing currently running Ollama models (ollama ps).
    
    Returns:
        Rich Table object with running models information
    """
    table = Table(show_header=True, box=None, padding=(0, 1), title="[bold]Running Models[/bold]")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Processor", style="green")
    
    try:
        ps_info = ollama_client.ps()
        if ps_info and 'models' in ps_info:
            models = ps_info['models']
            if models:
                for model_info in models:
                    model_name = model_info.get('name', 'unknown')
                    size = model_info.get('size', 0)
                    # Convert size to GB
                    size_gb = size / (1024**3) if size > 0 else 0
                    
                    # Get processor info (try different possible keys)
                    processor = "Unknown"
                    if 'details' in model_info:
                        details = model_info['details']
                        if 'parameter_size' in details:
                            processor = details.get('parameter_size', 'Unknown')
                    
                    table.add_row(
                        model_name,
                        f"{size_gb:.2f} GB",
                        processor
                    )
            else:
                table.add_row("[dim]No models loaded[/dim]", "", "")
        else:
            table.add_row("[dim]No models loaded[/dim]", "", "")
    except Exception as e:
        table.add_row(f"[red]Error: {str(e)}[/red]", "", "")
    
    return table


def run_benchmark_with_rich_layout(model_names: List[str], args) -> Dict[str, List[List[OllamaResponse]]]:
    """
    Executes benchmarks with Rich library side-by-side layout display.
    Shows streaming output on the left and live statistics on the right.
    
    Args:
        model_names: List of model names to benchmark
        args: Parsed command line arguments
        
    Returns:
        Dictionary mapping model names to lists of benchmark runs
    """
    console = Console()
    benchmarks: Dict[str, List[List[OllamaResponse]]] = {}
    
    for model_name in model_names:
        all_runs: List[List[OllamaResponse]] = []
        
        for run_number in range(args.runs):
            if args.runs > 1:
                console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
                console.print(f"[bold cyan]Run {run_number + 1}/{args.runs} for model: {model_name}[/bold cyan]")
                console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
            
            responses: List[OllamaResponse] = []
            
            for prompt_idx, prompt in enumerate(args.prompts):
                # Create layout for side-by-side display
                layout = Layout()
                layout.split_row(
                    Layout(name="output", ratio=2),
                    Layout(name="right_panel", ratio=1)
                )
                
                # Split right panel vertically 50/50
                layout["right_panel"].split_column(
                    Layout(name="stats"),
                    Layout(name="ollama_ps")
                )
                
                # Initialize stats data
                stats_data = {
                    "model_name": model_name,
                    "prompt_processing": 0.0,
                    "generation_speed": 0.0,
                    "combined_speed": 0.0,
                    "input_tokens": 0,
                    "generated_tokens": 0,
                    "load_time": 0.0,
                    "processing_time": 0.0,
                    "generation_time": 0.0,
                    "total_time": 0.0
                }
                
                # Initialize response text collector
                streamed_text = Text()
                max_lines_to_show = 50  # Keep last 50 lines visible
                
                with Live(layout, console=console, refresh_per_second=10, screen=True) as live:
                    # Show initial state
                    header = Text(f"Benchmarking: {model_name}\nPrompt {prompt_idx + 1}/{len(args.prompts)}: {prompt}\n\nResponse:\n", style="bold blue")
                    layout["output"].update(Panel(
                        header,
                        title="[bold cyan]Streaming Output[/bold cyan]",
                        border_style="blue"
                    ))
                    layout["stats"].update(Panel(
                        create_stats_table(**stats_data),
                        title="[bold cyan]Statistics[/bold cyan]",
                        border_style="cyan"
                    ))
                    layout["ollama_ps"].update(Panel(
                        create_ollama_ps_table(),
                        title="[bold cyan]Ollama PS[/bold cyan]",
                        border_style="yellow"
                    ))
                    
                    time.sleep(0.3)
                    
                    # Build options dict for GPU offloading if specified
                    options = {}
                    if args.num_gpu is not None:
                        options['num_gpu'] = args.num_gpu
                    
                    try:
                        # Stream the response
                        messages = [{"role": "user", "content": prompt}]
                        content = ""
                        stream = ollama_client.chat(
                            model=model_name,
                            messages=messages,
                            stream=True,
                            options=options if options else None,
                        )
                        
                        word_count = 0
                        for chunk in stream:
                            if hasattr(chunk.message, 'content'):
                                chunk_content = chunk.message.content
                                content += chunk_content
                                streamed_text.append(chunk_content)
                                
                                # Update output panel - show only last N lines to simulate scrolling
                                lines = streamed_text.plain.split('\n')
                                if len(lines) > max_lines_to_show:
                                    # Keep only the last max_lines_to_show lines
                                    visible_lines = '\n'.join(lines[-max_lines_to_show:])
                                    display_text = Text(visible_lines)
                                else:
                                    display_text = streamed_text
                                
                                output_content = Text(f"Benchmarking: {model_name}\nPrompt {prompt_idx + 1}/{len(args.prompts)}: {prompt}\n\nResponse:\n", style="bold blue")
                                output_content.append(display_text)
                                
                                layout["output"].update(Panel(
                                    output_content,
                                    title="[bold cyan]Streaming Output[/bold cyan]",
                                    border_style="blue"
                                ))
                                
                                # Update stats periodically (every 10 words)
                                word_count += len(chunk_content.split())
                                if word_count % 10 == 0:
                                    stats_data["generated_tokens"] = word_count
                                    layout["stats"].update(Panel(
                                        create_stats_table(**stats_data),
                                        title="[bold cyan]Statistics[/bold cyan]",
                                        border_style="cyan"
                                    ))
                                    layout["ollama_ps"].update(Panel(
                                        create_ollama_ps_table(),
                                        title="[bold cyan]Ollama PS[/bold cyan]",
                                        border_style="yellow"
                                    ))
                        
                        if not content.strip():
                            console.print(f"\n[bold red]Error: Ollama model {model_name} returned empty response.[/bold red]")
                            continue
                        
                        # Make a non-streaming call to get final metrics
                        response = ollama_client.chat(
                            model=model_name,
                            messages=messages,
                            options=options if options else None,
                        )
                        
                        # Create response object
                        benchmark_response = OllamaResponse(
                            model=model_name,
                            message=Message(
                                role="assistant",
                                content=content
                            ),
                            done=True,
                            total_duration=getattr(response, 'total_duration', 0),
                            load_duration=getattr(response, 'load_duration', 0),
                            prompt_eval_count=getattr(response, 'prompt_eval_count', 0),
                            prompt_eval_duration=getattr(response, 'prompt_eval_duration', 0),
                            eval_count=getattr(response, 'eval_count', 0),
                            eval_duration=getattr(response, 'eval_duration', 0)
                        )
                        
                        responses.append(benchmark_response)
                        
                        # Update final stats
                        stats_data["prompt_processing"] = benchmark_response.prompt_eval_count / nanosec_to_sec(benchmark_response.prompt_eval_duration) if benchmark_response.prompt_eval_duration > 0 else 0
                        stats_data["generation_speed"] = benchmark_response.eval_count / nanosec_to_sec(benchmark_response.eval_duration) if benchmark_response.eval_duration > 0 else 0
                        stats_data["combined_speed"] = (benchmark_response.prompt_eval_count + benchmark_response.eval_count) / nanosec_to_sec(benchmark_response.prompt_eval_duration + benchmark_response.eval_duration) if (benchmark_response.prompt_eval_duration + benchmark_response.eval_duration) > 0 else 0
                        stats_data["input_tokens"] = benchmark_response.prompt_eval_count
                        stats_data["generated_tokens"] = benchmark_response.eval_count
                        stats_data["load_time"] = nanosec_to_sec(benchmark_response.load_duration)
                        stats_data["processing_time"] = nanosec_to_sec(benchmark_response.prompt_eval_duration)
                        stats_data["generation_time"] = nanosec_to_sec(benchmark_response.eval_duration)
                        stats_data["total_time"] = nanosec_to_sec(benchmark_response.total_duration)
                        
                        layout["stats"].update(Panel(
                            create_stats_table(**stats_data),
                            title="[bold green]Statistics (Final)[/bold green]",
                            border_style="green"
                        ))
                        layout["ollama_ps"].update(Panel(
                            create_ollama_ps_table(),
                            title="[bold cyan]Ollama PS[/bold cyan]",
                            border_style="yellow"
                        ))
                        
                        time.sleep(2.0)  # Hold final view
                        
                    except Exception as e:
                        console.print(f"\n[bold red]Error benchmarking {model_name}: {str(e)}[/bold red]")
                        continue
            
            all_runs.append(responses)
            
            # Unload model from memory based on keep_model_loaded setting
            should_unload = False
            
            if not args.keep_model_loaded:
                should_unload = run_number < args.runs - 1 or model_names.index(model_name) < len(model_names) - 1
            elif model_names.index(model_name) < len(model_names) - 1 and run_number == args.runs - 1:
                should_unload = True
            
            if should_unload:
                try:
                    console.print(f"\n[yellow]Unloading {model_name} from memory...[/yellow]")
                    ollama_client.generate(model=model_name, prompt="", keep_alive=0)
                    console.print(f"[green]✓ {model_name} unloaded[/green]")
                except Exception as e:
                    console.print(f"[yellow]Note: Could not unload model: {e}[/yellow]")
        
        benchmarks[model_name] = all_runs
    
    return benchmarks


def run_benchmark_plain(model_names: List[str], args) -> Dict[str, List[List[OllamaResponse]]]:
    """
    Executes benchmarks with plain text output (original behavior).
    
    Args:
        model_names: List of model names to benchmark
        args: Parsed command line arguments
        
    Returns:
        Dictionary mapping model names to lists of benchmark runs
    """
    benchmarks: Dict[str, List[List[OllamaResponse]]] = {}
    
    # Execute benchmarks for each model and prompt
    for model_name in model_names:
        all_runs: List[List[OllamaResponse]] = []
        
        for run_number in range(args.runs):
            if args.runs > 1:
                print(f"\n{'='*60}\nRun {run_number + 1}/{args.runs} for model: {model_name}\n{'='*60}")
            
            responses: List[OllamaResponse] = []
            for prompt in args.prompts:
                if args.verbose:
                    print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")

                if response := run_benchmark(model_name, prompt, verbose=args.verbose, num_gpu=args.num_gpu):
                    responses.append(response)
                    if args.verbose:
                        print(f"Response: {response.message.content}")
                        inference_stats(response)
            
            all_runs.append(responses)
            
            # Unload model from memory based on keep_model_loaded setting
            should_unload = False
            
            if not args.keep_model_loaded:
                should_unload = run_number < args.runs - 1 or model_names.index(model_name) < len(model_names) - 1
            elif model_names.index(model_name) < len(model_names) - 1 and run_number == args.runs - 1:
                should_unload = True
            
            if should_unload:
                try:
                    print(f"\nUnloading {model_name} from memory...")
                    ollama_client.generate(model=model_name, prompt="", keep_alive=0)
                    print(f"✓ {model_name} unloaded")
                except Exception as e:
                    print(f"Note: Could not unload model: {e}")

        benchmarks[model_name] = all_runs
    
    return benchmarks


def main() -> None:
    """
    Main execution function for the benchmark tool.
    Handles argument parsing and orchestrates the benchmark process.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark performance metrics for Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output including streaming responses",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=[],
        help="Specific models to benchmark. Tests all available models if not specified.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
            # Short analytical question to test basic reasoning
            "Explain the process of photosynthesis in plants, including the key chemical reactions and energy transformations involved.",

            # Medium-length creative task
            "Write a detailed story about a time traveler who visits three different historical periods. Include specific details about each era and the protagonist's interactions.",

            # Long complex analysis
            "Analyze the potential impact of artificial intelligence on global employment over the next decade. Consider various industries, economic factors, and potential mitigation strategies. Provide specific examples and data-driven reasoning.",

            # Technical task with specific requirements
            "Write a Python function that implements a binary search tree with methods for insertion, deletion, and traversal. Include comments explaining the time complexity of each operation.",

            # Structured output task
            "Create a detailed business plan for a renewable energy startup. Include sections on market analysis, financial projections, competitive advantages, and risk assessment. Format the response with clear headings and bullet points.",
        ],
        help="Prompts to use for benchmarking. Multiple prompts can be specified. Default prompts test various capabilities including analysis, creativity, technical knowledge, and structured output.",
    )
    parser.add_argument(
        "-t",
        "--table",
        action="store_true",
        help="Output as table instead of separate results per model",
        default=False,
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help="Number of model layers to offload to GPU. Use this for large models that don't fit entirely in VRAM. Higher values use more GPU memory. Omit to let Ollama decide automatically.",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=3,
        help="Number of times to run each benchmark (default: 3). Results will show individual runs and averages.",
    )
    parser.add_argument(
        "-k",
        "--keep-model-loaded",
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=True,
        metavar='BOOL',
        help="Keep models loaded in memory between runs of the same model (default: True). Set to False to unload after each run for systems with limited VRAM or slow PCIe connections.",
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434). Can also be set via OLLAMA_HOST environment variable.",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=['rich', 'plain'],
        default='rich',
        help="Display layout style: 'rich' for side-by-side streaming output with live stats (default), 'plain' for classic output.",
    )

    args = parser.parse_args()
    
    # Initialize Ollama client with the specified host
    # Priority: CLI argument > environment variable > default
    global ollama_client
    host = os.getenv('OLLAMA_HOST', args.host)
    ollama_client = ollama.Client(host=host)
    
    print(
        f"\nOllama Host: {host}\nVerbose: {args.verbose}\nTest models: {args.models}\nPrompts: {args.prompts}\nTable Output: {args.table}\nRuns: {args.runs}\nKeep Model Loaded: {args.keep_model_loaded}\nLayout: {args.layout}"
    )

    model_names = get_benchmark_models(args.models)
    
    # Branch based on layout choice
    if args.layout == 'rich':
        benchmarks = run_benchmark_with_rich_layout(model_names, args)
    else:  # args.layout == 'plain'
        benchmarks = run_benchmark_plain(model_names, args)

    # Display final results
    if args.table:
        table_stats(benchmarks, args.runs)
    else:
        # Show summary statistics for all modes
        for model_name, all_runs in benchmarks.items():
            average_stats_multi_run(model_name, all_runs, args.runs)


if __name__ == "__main__":
    main()
