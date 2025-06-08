#!/usr/bin/env python3
"""
Enhanced CLI for Pro Mode System with UX improvements
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import readline  # For input history
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from pro_mode import ProModeSystem, ProModeConfig

# Initialize rich console
console = Console()

# History file for readline
HISTORY_FILE = Path.home() / ".pro_mode_history"

class ProModeCLI:
    """Enhanced CLI interface for Pro Mode System"""
    
    def __init__(self):
        self.config = None
        self.pro_mode = None
        self.session_history = []
        self.setup_readline()
    
    def setup_readline(self):
        """Setup readline for command history"""
        try:
            if HISTORY_FILE.exists():
                readline.read_history_file(HISTORY_FILE)
            readline.set_history_length(1000)
            # Enable bracketed paste mode for better multi-line handling
            readline.parse_and_bind('set enable-bracketed-paste on')
        except:
            pass
    
    def save_history(self):
        """Save command history"""
        try:
            readline.write_history_file(HISTORY_FILE)
        except:
            pass
    
    def display_banner(self):
        """Display welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                      Pro Mode System v1.0                      ║
║            Enhanced Multi-Perspective AI Responses             ║
╚═══════════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold cyan")
    
    def check_api_key(self) -> bool:
        """Check if API key is set"""
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]❌ OpenAI API key not found![/red]")
            console.print("\nPlease set your API key:")
            console.print("  export OPENAI_API_KEY='your-api-key-here'")
            return False
        return True
    
    def setup_config(self) -> ProModeConfig:
        """Interactive configuration setup"""
        console.print("\n[bold]Configuration Setup[/bold]")
        
        # Model selection
        models = ["o4-mini", "o3", "gpt-4.1"]
        console.print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            console.print(f"  {i}. {model}")
        
        model_choice = Prompt.ask(
            "Select model",
            choices=[str(i) for i in range(1, len(models) + 1)],
            default="1"
        )
        selected_model = models[int(model_choice) - 1]
        
        # Number of calls
        num_calls = Prompt.ask(
            "Number of intermediate calls",
            default="4"
        )
        
        config = ProModeConfig(
            model=selected_model,
            num_intermediate_calls=int(num_calls)
        )
        
        console.print(f"\n[green]✓ Configuration set:[/green]")
        console.print(f"  Model: {config.model}")
        console.print(f"  Intermediate calls: {config.num_intermediate_calls}")
        
        return config
    
    def display_results(self, result: Dict[str, Any], query: str):
        """Display results with enhanced formatting"""
        metadata = result['metadata']
        
        # Create metadata table
        table = Table(title="Processing Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Time", f"{metadata['total_time']:.2f}s")
        table.add_row("Intermediate Calls", str(metadata['num_intermediate_calls']))
        table.add_row("Intermediate Time", f"{metadata['intermediate_time']:.2f}s")
        table.add_row("Synthesis Time", f"{metadata['synthesis_time']:.2f}s")
        table.add_row("Total Tokens", str(metadata['total_usage']['total_tokens']))
        table.add_row("Prompt Tokens", str(metadata['total_usage']['prompt_tokens']))
        table.add_row("Completion Tokens", str(metadata['total_usage']['completion_tokens']))
        
        console.print("\n")
        console.print(table)
        
        # Display final answer
        console.print("\n[bold cyan]Final Answer:[/bold cyan]")
        answer_panel = Panel(
            Markdown(result['final_answer']),
            border_style="green",
            padding=(1, 2)
        )
        console.print(answer_panel)
        
        # Save to session history
        self.session_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result': result
        })
    
    def show_intermediate_responses(self, result: Dict[str, Any]):
        """Display intermediate responses"""
        console.print("\n[bold]Intermediate Responses:[/bold]")
        
        for i, response in enumerate(result['intermediate_responses'], 1):
            console.print(f"\n[cyan]Response {i}:[/cyan]")
            # Show first 500 chars of each response
            preview = response[:500] + "..." if len(response) > 500 else response
            console.print(Panel(preview, border_style="dim"))
    
    def save_session(self):
        """Save session history"""
        if not self.session_history:
            console.print("[yellow]No queries to save[/yellow]")
            return
        
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.session_history, f, indent=2)
        console.print(f"[green]✓ Session saved to {filename}[/green]")
    
    def process_query(self, query: str, show_intermediate: bool = False):
        """Process a single query with progress display"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Main task
            task = progress.add_task("Processing query...", total=3)
            
            try:
                # Step 1: Making intermediate calls
                progress.update(task, description="Making intermediate calls...")
                start_time = time.time()
                
                # Actually process the query
                result = self.pro_mode.process_input(query)
                
                progress.update(task, advance=2, description="Synthesizing responses...")
                time.sleep(0.5)  # Brief pause for visual effect
                
                progress.update(task, advance=1, description="Complete!")
                
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                return
        
        # Display results
        self.display_results(result, query)
        
        if show_intermediate:
            self.show_intermediate_responses(result)
    
    def get_multiline_input(self) -> str:
        """Get input that may span multiple lines"""
        lines = []
        console.print("[dim]Enter your query (press Ctrl+D or type '/end' on a new line to finish):[/dim]")
        
        try:
            while True:
                line = console.input("[bold green]❯[/bold green] " if not lines else "[bold green].[/bold green] ")
                if line.strip() == '/end':
                    break
                lines.append(line)
        except EOFError:
            # Ctrl+D pressed
            pass
        
        return '\n'.join(lines).strip()
    
    
    def interactive_mode(self):
        """Enhanced interactive mode"""
        self.display_banner()
        
        if not self.check_api_key():
            return
        
        # Quick setup or custom
        use_defaults = Confirm.ask("\nUse default settings? (o4-mini, 4 calls)", default=True)
        
        if use_defaults:
            self.config = ProModeConfig(model="o4-mini", num_intermediate_calls=4)
        else:
            self.config = self.setup_config()
        
        self.pro_mode = ProModeSystem(self.config)
        
        console.print("\n[bold]Interactive Mode[/bold]")
        console.print("Commands:")
        console.print("  [cyan]/help[/cyan]     - Show this help")
        console.print("  [cyan]/config[/cyan]   - Show current configuration")
        console.print("  [cyan]/history[/cyan]  - Show query history")
        console.print("  [cyan]/save[/cyan]     - Save session to file")
        console.print("  [cyan]/details[/cyan]  - Toggle showing intermediate responses")
        console.print("  [cyan]/multi[/cyan]    - Enter multi-line mode")
        console.print("  [cyan]/exit[/cyan]     - Exit the program")
        console.print("\n[dim]Tip: The CLI automatically detects multi-line input when you paste.[/dim]")
        console.print("Type your query or a command:\n")
        
        show_intermediate = False
        
        while True:
            try:
                # Get input with rich prompt
                query = console.input("[bold green]❯[/bold green] ")
                
                # Strip only trailing whitespace to preserve intentional leading spaces
                query = query.rstrip()
                
                if not query:
                    continue
                
                # Check if input contains newlines (multi-line paste)
                if '\n' in query:
                    console.print("[dim]Detected multi-line input[/dim]")
                    # Process the multi-line query directly
                    self.process_query(query, show_intermediate)
                    continue
                
                # Handle commands (single line)
                if query.startswith('/'):
                    command = query.lower().strip()
                    
                    if command == '/exit':
                        if Confirm.ask("\nSave session before exiting?"):
                            self.save_session()
                        console.print("[yellow]Goodbye![/yellow]")
                        break
                    
                    elif command == '/help':
                        console.print("\n[bold]Available Commands:[/bold]")
                        console.print("  /help     - Show this help")
                        console.print("  /config   - Show current configuration")
                        console.print("  /history  - Show query history")
                        console.print("  /save     - Save session to file")
                        console.print("  /details  - Toggle showing intermediate responses")
                        console.print("  /multi    - Enter multi-line mode")
                        console.print("  /exit     - Exit the program\n")
                    
                    elif command == '/config':
                        console.print(f"\n[bold]Current Configuration:[/bold]")
                        console.print(f"  Model: {self.config.model}")
                        console.print(f"  Intermediate calls: {self.config.num_intermediate_calls}\n")
                    
                    elif command == '/history':
                        if not self.session_history:
                            console.print("[yellow]No queries in this session[/yellow]")
                        else:
                            console.print("\n[bold]Query History:[/bold]")
                            for i, item in enumerate(self.session_history, 1):
                                timestamp = datetime.fromisoformat(item['timestamp'])
                                console.print(f"{i}. [{timestamp.strftime('%H:%M:%S')}] {item['query'][:50]}...")
                    
                    elif command == '/save':
                        self.save_session()
                    
                    elif command == '/details':
                        show_intermediate = not show_intermediate
                        state = "ON" if show_intermediate else "OFF"
                        console.print(f"[yellow]Intermediate responses: {state}[/yellow]")
                    
                    elif command == '/multi':
                        query = self.get_multiline_input()
                        if query:
                            self.process_query(query, show_intermediate)
                        continue
                    
                    else:
                        console.print(f"[red]Unknown command: {command}[/red]")
                    
                    continue
                
                # Process the single-line query
                self.process_query(query.strip(), show_intermediate)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
        
        self.save_history()
    
    def batch_mode(self, queries_file: str, output_dir: str = "results"):
        """Process multiple queries from a file"""
        if not self.check_api_key():
            return
        
        # Setup config
        self.config = ProModeConfig()
        self.pro_mode = ProModeSystem(self.config)
        
        # Read queries
        try:
            with open(queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        except Exception as e:
            console.print(f"[red]Error reading file: {str(e)}[/red]")
            return
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        console.print(f"\n[bold]Processing {len(queries)} queries[/bold]")
        
        # Process each query
        for i, query in enumerate(queries, 1):
            console.print(f"\n[cyan]Query {i}/{len(queries)}:[/cyan] {query[:50]}...")
            
            try:
                result = self.pro_mode.process_input(query)
                
                # Save result
                output_file = Path(output_dir) / f"query_{i:03d}.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        'query': query,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                
                console.print(f"[green]✓ Saved to {output_file}[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ Error: {str(e)}[/red]")
        
        console.print(f"\n[green]Batch processing complete! Results in {output_dir}/[/green]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pro Mode CLI - Enhanced multi-perspective AI responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive mode
  %(prog)s -q "Your query"    # Single query
  %(prog)s -b queries.txt     # Batch mode
  %(prog)s -q "Query" -d      # Show intermediate responses
        """
    )
    
    parser.add_argument('-q', '--query', type=str, help='Single query to process')
    parser.add_argument('-b', '--batch', type=str, help='File with queries (one per line)')
    parser.add_argument('-o', '--output', type=str, help='Output directory for batch mode')
    parser.add_argument('-d', '--details', action='store_true', help='Show intermediate responses')
    parser.add_argument('-m', '--model', type=str, default='gpt-4o-mini', help='Model to use')
    parser.add_argument('-c', '--calls', type=int, default=4, help='Number of intermediate calls')
    
    args = parser.parse_args()
    
    cli = ProModeCLI()
    
    if args.query:
        # Single query mode
        if not cli.check_api_key():
            sys.exit(1)
        
        cli.config = ProModeConfig(model=args.model, num_intermediate_calls=args.calls)
        cli.pro_mode = ProModeSystem(cli.config)
        
        console.print(f"\n[bold]Processing query...[/bold]")
        cli.process_query(args.query, args.details)
        
    elif args.batch:
        # Batch mode
        output_dir = args.output or "results"
        cli.batch_mode(args.batch, output_dir)
        
    else:
        # Interactive mode
        cli.interactive_mode()


if __name__ == "__main__":
    main()
