# LLM Benchmark tool for Ollama

This tool allows you to get the t/s (tokens per second) of Large Language Models (LLMs) running on your local machine.

## Example output

Output on a Nvidia 4090 windows desktop

```bash
Average stats:
(Running on dual 3090 Ti GPU, Epyc 7763 CPU in Ubuntu 22.04)

----------------------------------------------------
        Model: deepseek-r1:70b
        Performance Metrics:
            Prompt Processing:  336.73 tokens/sec
            Generation Speed:   17.65 tokens/sec
            Combined Speed:     18.01 tokens/sec

        Workload Stats:
            Input Tokens:       165
            Generated Tokens:   7673
            Model Load Time:    6.11s
            Processing Time:    0.49s
            Generation Time:    434.70s
            Total Time:         441.31s
----------------------------------------------------

Average stats: 
(Running on single 3090 GPU, 13900KS CPU in WSL2(Ubuntu 22.04) in Windows 11)

----------------------------------------------------
        Model: deepseek-r1:32b
        Performance Metrics:
            Prompt Processing:  399.05 tokens/sec
            Generation Speed:   27.18 tokens/sec
            Combined Speed:     27.58 tokens/sec

        Workload Stats:
            Input Tokens:       168
            Generated Tokens:   10601
            Model Load Time:    15.44s
            Processing Time:    0.42s
            Generation Time:    390.00s
            Total Time:         405.87s
----------------------------------------------------
```

## Getting Started

Follow these instructions to set up and run benchmarks on your system.

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.com/) installed and configured

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/jammsen/ollama-benchmark.git
   cd ollama-benchmark
   ```

2. **Set up Python environment**

   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run Benchmarks**

   Ensure that Ollama is running:

   Basic usage:

   ```bash
   python benchmark.py
   ```

   Singe model & verbose option:

   ```bash
   python benchmark.py --verbose --models deepseek-r1:70b --prompts "Write a hello world program" "Explain quantum computing"
   ```

   Multi model, verbose, table and big prompt option:

   ```bash
   python benchmark.py --verbose --models llama3 gemma3 gpt-oss --prompts "Design a scalable web application architecture for an e-commerce platform that needs to handle 10,000 concurrent users during peak shopping periods. Your architecture should include a detailed database design with proper normalization, effective caching strategies using Redis or Memcached, load balancers, CDN integration, payment processing systems, inventory management, user authentication and authorization, session management, and monitoring tools. Explain the key components, their interactions, data flow between services, and how the system maintains performance and reliability under high traffic loads while ensuring secure transactions." -t
   ```

### Command Line Options

- `-v, --verbose`: Enable detailed output including streaming responses
- `-m, --models`: Space-separated list of models to benchmark (defaults to all available models)
- `-p, --prompts`: Space-separated list of custom prompts (defaults to a predefined set testing various capabilities)
- `-t, --table_output`: Results printed into a table output
- `--num-gpu`: Number of model layers to offload to GPU (useful for large models that don't fit entirely in VRAM)

#### GPU Layer Offloading

For large models that exceed your GPU's VRAM capacity, you can use the `--num-gpu` parameter to control how many layers are loaded on the GPU:

```bash
# Load 35 layers on GPU, rest on CPU
python benchmark.py --verbose --models gpt-oss:120b -p "hello there" --num-gpu 35
```

This is useful for running models like 70B or 120B parameter models on consumer GPUs with limited VRAM. Higher values use more GPU memory but provide better performance.

#### Remote Ollama Instances

To benchmark against a remote Ollama instance or a Docker container, set the `OLLAMA_HOST` environment variable:

```bash
# Connect to Ollama running on a different host
OLLAMA_HOST=http://192.168.1.100:11434 python benchmark.py --models llama3

# Connect to Ollama running in Docker
OLLAMA_HOST=http://localhost:11434 python benchmark.py --models llama3
```

### Default Benchmark Suite

The default benchmark suite includes prompts testing:

- Analytical reasoning
- Creative writing
- Complex analysis
- Technical knowledge
- Structured output generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
