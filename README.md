# KV-Cache Playground

An interactive Streamlit application for understanding and visualizing Key-Value (KV) caching in Transformer models.

## Features

- **Interactive Demo**: Compare text generation with and without KV-cache
- **Performance Benchmarks**: Visualize speed improvements across different token counts
- **Educational Visualizations**: Understand how KV-cache reduces computational complexity
- **Experiment Playground**: Calculate memory usage and estimate speedups
- **Sleek Modern UI**: Beautiful gradient-based design with glass morphism effects

## Prerequisites

- Python 3.11+
- uv (for package management)

## Installation

1. **Clone the repository**:
```bash
cd /path/to/kv-cache-playground
```

2. **Create and activate virtual environment**:
```bash
uv venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
uv pip install streamlit torch numpy pandas plotly matplotlib transformers requests tqdm
```

## Running the Playground

```bash
streamlit run kv_cache_app.py
```

The app will open in your default browser at `http://localhost:8501`

## How It Works

### KV-Cache Explained

Key-Value caching is an optimization technique for autoregressive generation in Transformers:

- **Without Cache**: Recomputes keys and values for the entire sequence at each generation step (O(n²) complexity)
- **With Cache**: Stores previously computed keys and values, only computing for new tokens (O(n) complexity)

### App Structure

- `kv_cache_app.py`: Main Streamlit application
- `models.py`: PyTorch implementation of Transformer models with and without KV-cache
- `visualizations.py`: Plotly-based interactive charts and graphs
- `ui_components.py`: Custom CSS and UI components for modern design
- `kv_cache_tutorial.py`: Original tutorial implementation for reference

## Usage Guide

1. **Interactive Demo Tab**: 
   - Enter a text prompt
   - Adjust token generation count
   - Click "Generate Text" to see side-by-side comparison

2. **Benchmarks Tab**:
   - Select token counts to benchmark
   - Run benchmarks to see performance graphs
   - View speedup metrics

3. **Understanding Tab**:
   - Visual explanation of KV-cache concepts
   - Interactive complexity analysis
   - Computation pattern visualization

4. **Playground Tab**:
   - Calculate memory requirements for different configurations
   - Estimate speedups based on model parameters
   - Experiment with various settings

## Configuration

Adjust model parameters in the sidebar:
- **Context Length**: Maximum sequence length (32-256)
- **Embedding Size**: Model hidden dimension (32-256)
- **Number of Heads**: Multi-head attention heads (1, 2, 4, 8)
- **Number of Layers**: Transformer blocks (1-4)

## Performance Notes

- The playground uses CPU-only computation for consistent benchmarking
- Actual speedups vary based on sequence length and model size
- Larger models and longer sequences show more dramatic improvements

## Contributing

Feel free to submit issues or pull requests to improve the playground!

## License

MIT License