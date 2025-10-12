import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import math
from pathlib import Path
import requests
import pandas as pd
from models import BaselineModel, KVCacheModel
from visualizations import plot_attention_heatmap, create_benchmark_chart, create_speedup_chart
from ui_components import apply_custom_css, show_header

# Force CPU usage
device = 'cpu'
torch.set_num_threads(1)

# Page configuration
st.set_page_config(
    page_title="KV-Cache Playground",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state
if 'model_params' not in st.session_state:
    st.session_state.model_params = {
        'vocab_size': 65,
        'block_size': 128,
        'embed_size': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.0
    }

if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = None

# Data preparation
@st.cache_data
def load_shakespeare_data():
    """Load and prepare the Tiny Shakespeare dataset."""
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    data_path = Path('input.txt')
    
    if not data_path.exists():
        r = requests.get(data_url, timeout=30)
        r.raise_for_status()
        data_path.write_text(r.text, encoding='utf-8')
    
    text = data_path.read_text(encoding='utf-8')
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    return text, vocab, stoi, itos

def main():
    # Show header
    show_header()
    # Load data
    text, vocab, stoi, itos = load_shakespeare_data()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Model Configuration")
        
        with st.expander("Architecture Parameters", expanded=True):
            block_size = st.slider("Context Length", 32, 256, st.session_state.model_params['block_size'], 32)
            embed_size = st.slider("Embedding Size", 32, 256, st.session_state.model_params['embed_size'], 32)
            num_heads = st.select_slider("Number of Heads", [1, 2, 4, 8], st.session_state.model_params['num_heads'])
            num_layers = st.slider("Number of Layers", 1, 4, st.session_state.model_params['num_layers'])
            
            st.session_state.model_params.update({
                'block_size': block_size,
                'embed_size': embed_size,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'vocab_size': len(vocab),
                'dropout': 0.0
            })
        
        st.markdown("### About KV-Cache")
        with st.expander("Learn More"):
            st.markdown("""
            **Key-Value Caching** is an optimization that:
            - **Stores** previously computed K/V vectors
            - **Reuses** them in subsequent steps
            - **Reduces** complexity from O(n²) to O(n)
            - **Trades** memory for speed
            """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Interactive Demo", "Benchmarks", "Understanding", "Playground"])
    
    with tab1:
        st.markdown("### Generate Text Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prompt = st.text_input("Enter prompt:", value="To be or not to be")
            max_tokens = st.slider("Tokens to generate:", 10, 200, 50)
        
        with col2:
            temperature = st.slider("Temperature:", 0.1, 2.0, 1.0, 0.1)
            st.info("Higher temperature = more random")
        
        if st.button("Generate Text", use_container_width=True):
            with st.spinner("Generating..."):
                # Initialize models
                baseline_model = BaselineModel(**st.session_state.model_params).to(device)
                kvcache_model = KVCacheModel(**st.session_state.model_params).to(device)
                
                # Share weights
                kvcache_model.load_state_dict(baseline_model.state_dict(), strict=False)
                
                # Encode prompt
                prompt_encoded = [stoi[c] for c in prompt if c in stoi]
                if not prompt_encoded:
                    prompt_encoded = [0]
                x = torch.tensor([prompt_encoded], dtype=torch.long, device=device)
                
                # Generate with timing
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Without KV-Cache")
                    t0 = time.time()
                    output_baseline = baseline_model.generate(x.clone(), max_tokens)
                    time_baseline = time.time() - t0
                    text_baseline = ''.join([itos[i] for i in output_baseline[0].tolist()])
                    
                    st.text_area("Output:", text_baseline, height=150, key="baseline")
                    st.metric("Time", f"{time_baseline:.3f}s")
                    st.metric("Speed", f"{max_tokens/time_baseline:.1f} tok/s")
                
                with col2:
                    st.markdown("#### With KV-Cache")
                    t0 = time.time()
                    output_cached = kvcache_model.generate_cached(x.clone(), max_tokens)
                    time_cached = time.time() - t0
                    text_cached = ''.join([itos[i] for i in output_cached[0].tolist()])
                    
                    st.text_area("Output:", text_cached, height=150, key="cached")
                    st.metric("Time", f"{time_cached:.3f}s")
                    st.metric("Speed", f"{max_tokens/time_cached:.1f} tok/s")
                
                # Speedup
                speedup = time_baseline / time_cached if time_cached > 0 else 0
                st.success(f"**{speedup:.2f}x Speedup** with KV-Cache!")
    
    with tab2:
        st.markdown("### Performance Benchmarks")
        
        token_counts = st.multiselect(
            "Token counts to benchmark:",
            [25, 50, 100, 200, 300],
            default=[50, 100, 200]
        )
        
        if st.button("Run Benchmark", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            results = []
            baseline = BaselineModel(**st.session_state.model_params).to(device)
            cached = KVCacheModel(**st.session_state.model_params).to(device)
            cached.load_state_dict(baseline.state_dict(), strict=False)
            
            for i, tokens in enumerate(token_counts):
                status.text(f"Testing {tokens} tokens...")
                progress.progress((i + 1) / len(token_counts))
                
                x = torch.tensor([[0]], dtype=torch.long, device=device)
                
                t0 = time.time()
                _ = baseline.generate(x.clone(), tokens)
                t_baseline = time.time() - t0
                
                t0 = time.time()
                _ = cached.generate_cached(x.clone(), tokens)
                t_cached = time.time() - t0
                
                results.append((tokens, t_baseline, t_cached))
            
            st.session_state.benchmark_results = results
            progress.empty()
            status.empty()
        
        if st.session_state.benchmark_results:
            fig = create_benchmark_chart(st.session_state.benchmark_results)
            st.plotly_chart(fig, use_container_width=True)
            
            df = pd.DataFrame(
                st.session_state.benchmark_results,
                columns=['Tokens', 'No Cache (s)', 'With Cache (s)']
            )
            df['Speedup'] = (df['No Cache (s)'] / df['With Cache (s)']).round(2).astype(str) + 'x'
            st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.markdown("### How KV-Cache Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("**Without KV-Cache**")
            st.markdown("""
            - Recompute K,V for entire sequence
            - Time: O(n²) per generation
            - Memory: Minimal
            - Slower for long sequences
            """)
            
            fig = create_speedup_chart(False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.success("**With KV-Cache**")
            st.markdown("""
            - Store previous K,V vectors
            - Time: O(n) per generation
            - Memory: Stores past states
            - Much faster for long sequences
            """)
            
            fig = create_speedup_chart(True)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Interactive Complexity Analysis")
        
        seq_len = st.slider("Sequence Length", 10, 500, 100, 10)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            compute_no_cache = seq_len ** 2
            st.metric("Computations (No Cache)", f"{compute_no_cache:,}")
        
        with col2:
            compute_cache = seq_len
            st.metric("Computations (With Cache)", f"{compute_cache:,}")
        
        with col3:
            reduction = (1 - compute_cache/compute_no_cache) * 100
            st.metric("Computation Reduction", f"{reduction:.1f}%")
    
    with tab4:
        st.markdown("### Experiment Playground")
        
        st.info("Try different configurations and observe the impact on performance!")
        
        # Memory usage calculator
        st.markdown("#### Memory Usage Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch = st.number_input("Batch Size", 1, 32, 1)
            seq = st.number_input("Sequence Length", 128, 2048, 512)
            hidden = st.number_input("Hidden Size", 128, 1024, 256)
            layers = st.number_input("Layers", 1, 24, 6)
        
        with col2:
            # Calculate memory
            kv_per_token = 2 * hidden * 4 / (1024**2)  # MB (float32)
            total_kv = batch * seq * layers * kv_per_token
            
            st.metric("KV-Cache Memory", f"{total_kv:.1f} MB")
            st.metric("Per Token", f"{kv_per_token:.3f} MB")
            
            if total_kv > 1024:
                st.warning(f"Large cache: {total_kv/1024:.1f} GB")
        
        # Speed estimator
        st.markdown("#### Speed Estimator")
        
        tokens_to_gen = st.slider("Tokens to Generate", 100, 1000, 500)
        
        # Estimate based on model size
        flops_per_token_nocache = seq * hidden * hidden * layers
        flops_per_token_cache = hidden * hidden * layers
        
        speedup_est = flops_per_token_nocache / flops_per_token_cache
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Estimated Speedup", f"{speedup_est:.1f}x")
        
        with col2:
            time_saved = tokens_to_gen * (speedup_est - 1) * 0.001  # Rough estimate
            st.metric("Time Saved", f"~{time_saved:.1f}s")
        
        with col3:
            efficiency = (speedup_est / seq) * 100
            st.metric("Cache Efficiency", f"{min(efficiency, 100):.0f}%")

if __name__ == "__main__":
    main()
