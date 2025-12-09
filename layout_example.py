#!/usr/bin/env python3
"""
Example layout demonstrating side-by-side display for benchmark results.
This shows how streaming output and statistics could be displayed together.
"""

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import time
import random


def create_stats_table(model_name: str, prompt_processing: float, generation_speed: float, 
                       combined_speed: float, input_tokens: int, generated_tokens: int,
                       load_time: float, processing_time: float, generation_time: float,
                       total_time: float) -> Table:
    """Create a statistics table for the benchmark results."""
    table = Table(title=f"Model: {model_name}", show_header=False, border_style="cyan")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="green")
    
    table.add_row("Performance Metrics", "")
    table.add_row("  Prompt Processing", f"{prompt_processing:.2f} tokens/sec")
    table.add_row("  Generation Speed", f"{generation_speed:.2f} tokens/sec")
    table.add_row("  Combined Speed", f"{combined_speed:.2f} tokens/sec")
    table.add_row("", "")
    table.add_row("Workload Stats", "")
    table.add_row("  Input Tokens", str(input_tokens))
    table.add_row("  Generated Tokens", str(generated_tokens))
    table.add_row("  Model Load Time", f"{load_time:.2f}s")
    table.add_row("  Processing Time", f"{processing_time:.2f}s")
    table.add_row("  Generation Time", f"{generation_time:.2f}s")
    table.add_row("  Total Time", f"{total_time:.2f}s")
    
    return table


def simulate_streaming_with_stats():
    """Simulate streaming output with live statistics display side-by-side."""
    console = Console()
    
    # Mock data
    model_name = "qwen3:4b"
    prompt = "Hello there! How can I help you today?"
    # Much longer response for 15 second demo
    response_text = """Hello there! 👋 How can I help you today? I'm here to assist you with any questions or tasks you might have. 
    
Whether you need help with coding problems, want to discuss various topics, need advice on a project, or just want to have a conversation, I'm ready to help! 

I can assist with a wide range of tasks including:
- Answering questions and providing explanations
- Helping with programming and debugging code
- Offering advice and recommendations
- Discussing various topics from science to philosophy
- Assisting with creative writing and brainstorming
- And much more!

Feel free to ask me anything, and I'll do my best to provide helpful and accurate information. What would you like to talk about or work on today?"""
    
    # Mock statistics (these would update in real benchmark)
    stats_data = {
        "model_name": model_name,
        "prompt_processing": 977.43,
        "generation_speed": 93.51,
        "combined_speed": 96.75,
        "input_tokens": 12,
        "generated_tokens": 0,
        "load_time": 0.07,
        "processing_time": 0.01,
        "generation_time": 0.0,
        "total_time": 0.0
    }
    
    # Create layout
    layout = Layout()
    layout.split_row(
        Layout(name="output", ratio=2),
        Layout(name="stats", ratio=1)
    )
    
    # Initialize response text collector
    streamed_text = Text()
    
    with Live(layout, console=console, refresh_per_second=10, screen=True) as live:
        # Show initial state
        layout["output"].update(Panel(
            Text(f"Benchmarking: {model_name}\nPrompt: {prompt}\n\nResponse:\n", style="bold blue"),
            title="[bold cyan]Streaming Output[/bold cyan]",
            border_style="blue"
        ))
        layout["stats"].update(Panel(
            create_stats_table(**stats_data),
            title="[bold cyan]Statistics[/bold cyan]",
            border_style="cyan"
        ))
        
        time.sleep(1.0)
        
        # Simulate streaming response word by word
        words = response_text.split()
        total_words = len(words)
        
        for i, word in enumerate(words):
            streamed_text.append(word + " ")
            
            # Update output panel with accumulated text
            output_content = Text(f"Benchmarking: {model_name}\nPrompt: {prompt}\n\nResponse:\n", style="bold blue")
            output_content.append(streamed_text)
            
            layout["output"].update(Panel(
                output_content,
                title="[bold cyan]Streaming Output[/bold cyan]",
                border_style="blue"
            ))
            
            # Update stats every 3 words for more realistic updates
            if i % 3 == 0 and i > 0:
                elapsed_time = (i / total_words) * 15.0  # Simulate 15 seconds total
                stats_data["generated_tokens"] = i
                stats_data["generation_time"] = elapsed_time
                stats_data["total_time"] = 0.08 + elapsed_time
                stats_data["generation_speed"] = stats_data["generated_tokens"] / stats_data["generation_time"] if stats_data["generation_time"] > 0 else 0
                stats_data["combined_speed"] = (stats_data["input_tokens"] + stats_data["generated_tokens"]) / (stats_data["processing_time"] + stats_data["generation_time"]) if (stats_data["processing_time"] + stats_data["generation_time"]) > 0 else 0
                
                layout["stats"].update(Panel(
                    create_stats_table(**stats_data),
                    title="[bold cyan]Statistics[/bold cyan]",
                    border_style="cyan"
                ))
            
            # Slow down to make it ~15 seconds total
            time.sleep(15.0 / total_words)
        
        # Final update with complete stats
        stats_data["generated_tokens"] = total_words
        stats_data["generation_time"] = 15.0
        stats_data["total_time"] = 15.08
        stats_data["generation_speed"] = stats_data["generated_tokens"] / stats_data["generation_time"]
        stats_data["combined_speed"] = (stats_data["input_tokens"] + stats_data["generated_tokens"]) / (stats_data["processing_time"] + stats_data["generation_time"])
        
        layout["stats"].update(Panel(
            create_stats_table(**stats_data),
            title="[bold cyan]Statistics (Final)[/bold cyan]",
            border_style="green"
        ))
        
        time.sleep(3.0)  # Hold final view for 3 seconds
        console.print("\n[bold green]✓ Benchmark complete![/bold green]\n")


def show_multiple_runs_table():
    """Show example of table output for multiple runs."""
    console = Console()
    
    console.print("\n[bold cyan]Example: Table Output for Multiple Runs[/bold cyan]\n")
    
    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Model\nName", style="cyan")
    table.add_column("Run", style="yellow")
    table.add_column("Prompt\nEval Rate\n(T/s)", justify="right", style="green")
    table.add_column("Generation\nRate\n(T/s)", justify="right", style="green")
    table.add_column("Total\nRate\n(T/s)", justify="right", style="green")
    table.add_column("Load Time\n(s)", justify="right")
    table.add_column("Total Time\n(s)", justify="right")
    
    # Mock data for multiple runs
    models = ["qwen3:4b", "llama3.2:1b"]
    
    for model in models:
        for run in range(1, 4):
            # Generate slightly varying mock data
            prompt_rate = 900 + random.randint(-50, 50)
            gen_rate = 90 + random.randint(-10, 10)
            total_rate = 95 + random.randint(-5, 5)
            load_time = round(random.uniform(0.05, 0.15), 2)
            total_time = round(random.uniform(3.0, 4.0), 2)
            
            table.add_row(
                model,
                f"Run {run}",
                f"{prompt_rate:.2f}",
                f"{gen_rate:.2f}",
                f"{total_rate:.2f}",
                f"{load_time:.2f}",
                f"{total_time:.2f}"
            )
        
        # Add average row
        table.add_row(
            model,
            "[bold]Average[/bold]",
            "[bold]920.50[/bold]",
            "[bold]92.15[/bold]",
            "[bold]95.80[/bold]",
            "[bold]0.09[/bold]",
            "[bold]3.45[/bold]",
            style="bold"
        )
    
    console.print(table)


def main():
    """Main function to demonstrate different layout options."""
    console = Console()
    
    console.print("\n[bold magenta]═══════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  Benchmark Layout Examples[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════[/bold magenta]\n")
    
    console.print("[bold yellow]Example 1: Side-by-side streaming output with live statistics[/bold yellow]\n")
    simulate_streaming_with_stats()
    
    console.print("\n" + "─" * 80 + "\n")
    
    console.print("[bold yellow]Example 2: Compact table output for multiple runs[/bold yellow]")
    show_multiple_runs_table()
    
    console.print("\n[bold green]Layout examples complete![/bold green]")
    console.print("[italic]These layouts could be integrated into the benchmark tool using the 'rich' library.[/italic]\n")


if __name__ == "__main__":
    main()
