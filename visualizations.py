import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_attention_heatmap(attention_weights, layer_idx=0, head_idx=0):
    """Create an interactive attention heatmap using Plotly."""
    if attention_weights and len(attention_weights) > layer_idx:
        layer_weights = attention_weights[layer_idx]
        if len(layer_weights) > head_idx:
            weights = layer_weights[head_idx][0, :, :].detach().numpy()
            
            fig = go.Figure(data=go.Heatmap(
                z=weights,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Attention",
                    titleside="right",
                    tickmode="linear",
                    tick0=0,
                    dtick=0.2
                )
            ))
            
            fig.update_layout(
                title=f"Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}",
                xaxis_title="Key Position",
                yaxis_title="Query Position",
                height=400,
                template="plotly_white",
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='#333333')
            )
            
            return fig
    return None

def create_benchmark_chart(results):
    """Create an interactive benchmark comparison chart."""
    if not results:
        return None
    
    df = pd.DataFrame(results, columns=['Tokens', 'No Cache (s)', 'With Cache (s)'])
    
    fig = go.Figure()
    
    # Without cache line
    fig.add_trace(go.Scatter(
        x=df['Tokens'],
        y=df['No Cache (s)'],
        mode='lines+markers',
        name='Without KV-Cache',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10, color='#FF6B6B'),
        hovertemplate='<b>Without Cache</b><br>Tokens: %{x}<br>Time: %{y:.3f}s<extra></extra>'
    ))
    
    # With cache line
    fig.add_trace(go.Scatter(
        x=df['Tokens'],
        y=df['With Cache (s)'],
        mode='lines+markers',
        name='With KV-Cache',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=10, color='#4ECDC4'),
        hovertemplate='<b>With Cache</b><br>Tokens: %{x}<br>Time: %{y:.3f}s<extra></extra>'
    ))
    
    # Add speedup annotations
    for _, row in df.iterrows():
        speedup = row['No Cache (s)'] / row['With Cache (s)']
        fig.add_annotation(
            x=row['Tokens'],
            y=row['With Cache (s)'],
            text=f"{speedup:.1f}x",
            showarrow=False,
            yshift=15,
            font=dict(color='#4ECDC4', size=12, family='monospace'),
            bgcolor='rgba(78, 205, 196, 0.1)',
            bordercolor='#4ECDC4',
            borderwidth=1,
            borderpad=4
        )
    
    fig.update_layout(
        title="Generation Time Comparison",
        xaxis_title="Number of Tokens Generated",
        yaxis_title="Time (seconds)",
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333333', size=12),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor='#e9ecef',
            borderwidth=1,
            font=dict(size=12)
        ),
        xaxis=dict(
            gridcolor='#f8f9fa',
            zerolinecolor='#e9ecef'
        ),
        yaxis=dict(
            gridcolor='#f8f9fa',
            zerolinecolor='#e9ecef'
        )
    )
    
    return fig

def create_speedup_chart(with_cache=False):
    """Create a chart showing computation pattern."""
    n_steps = 5
    fig = go.Figure()
    
    if with_cache:
        # Constant computation with cache
        fig.add_trace(go.Bar(
            x=[f"Step {i+1}" for i in range(n_steps)],
            y=[1] * n_steps,
            marker=dict(
                color=['#4ECDC4'] * n_steps,
                line=dict(color='#3BA99C', width=2)
            ),
            text=['O(n)'] * n_steps,
            textposition='outside',
            name='With Cache'
        ))
        title = "Constant Computation with Cache"
    else:
        # Growing computation without cache
        fig.add_trace(go.Bar(
            x=[f"Step {i+1}" for i in range(n_steps)],
            y=[i+1 for i in range(n_steps)],
            marker=dict(
                color=['#FF6B6B'] * n_steps,
                line=dict(color='#E85555', width=2)
            ),
            text=[f'O(n²)' if i == n_steps-1 else '' for i in range(n_steps)],
            textposition='outside',
            name='Without Cache'
        ))
        title = "Growing Computation without Cache"
    
    fig.update_layout(
        title=title,
        yaxis_title="Computation Units",
        xaxis_title="Generation Step",
        showlegend=False,
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=300,
        font=dict(color='#333333'),
        yaxis=dict(
            gridcolor='#f8f9fa',
            zerolinecolor='#e9ecef'
        ),
        xaxis=dict(
            gridcolor='#f8f9fa'
        )
    )
    
    return fig

def create_memory_usage_chart(batch_size, seq_length, hidden_size, num_layers):
    """Create a memory usage breakdown chart."""
    # Calculate memory components (in MB)
    kv_per_layer = 2 * batch_size * seq_length * hidden_size * 4 / (1024**2)
    total_kv = kv_per_layer * num_layers
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Keys', 'Values', 'Other'],
        values=[total_kv/2, total_kv/2, total_kv*0.1],
        hole=0.3,
        marker=dict(
            colors=['#667eea', '#764ba2', '#f093fb'],
            line=dict(color='white', width=2)
        )
    )])
    
    fig.update_layout(
        title=f"KV-Cache Memory Breakdown ({total_kv:.1f} MB total)",
        template="plotly_white",
        paper_bgcolor='white',
        font=dict(color='#333333'),
        showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor='#e9ecef',
            borderwidth=1
        )
    )
    
    return fig

def create_complexity_comparison(max_length=100):
    """Create a complexity comparison chart."""
    lengths = np.arange(1, max_length+1)
    no_cache = lengths ** 2
    with_cache = lengths
    
    fig = go.Figure()
    
    # Without cache - quadratic
    fig.add_trace(go.Scatter(
        x=lengths,
        y=no_cache,
        mode='lines',
        name='Without Cache O(n²)',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.1)'
    ))
    
    # With cache - linear
    fig.add_trace(go.Scatter(
        x=lengths,
        y=with_cache,
        mode='lines',
        name='With Cache O(n)',
        line=dict(color='#4ECDC4', width=2),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.1)'
    ))
    
    fig.update_layout(
        title="Computational Complexity Comparison",
        xaxis_title="Sequence Length",
        yaxis_title="Relative Computation",
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333333'),
        hovermode='x unified',
        legend=dict(
            bgcolor='white',
            bordercolor='#e9ecef',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='#f8f9fa',
            zerolinecolor='#e9ecef'
        ),
        yaxis=dict(
            gridcolor='#f8f9fa',
            zerolinecolor='#e9ecef',
            type='log'
        )
    )
    
    return fig

def create_speedup_heatmap(seq_lengths, model_sizes):
    """Create a heatmap showing speedup across different configurations."""
    # Calculate theoretical speedups
    speedups = np.zeros((len(seq_lengths), len(model_sizes)))
    
    for i, seq_len in enumerate(seq_lengths):
        for j, model_size in enumerate(model_sizes):
            # Simplified speedup calculation
            speedups[i, j] = min(seq_len / 2, model_size / 100)
    
    fig = go.Figure(data=go.Heatmap(
        z=speedups,
        x=[f"{s}M" for s in model_sizes],
        y=[f"{l}" for l in seq_lengths],
        colorscale='Viridis',
        text=np.round(speedups, 1),
        texttemplate="%{text}x",
        textfont={"size": 10},
        colorbar=dict(title="Speedup")
    ))
    
    fig.update_layout(
        title="KV-Cache Speedup Heatmap",
        xaxis_title="Model Size (Parameters)",
        yaxis_title="Sequence Length",
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333333')
    )
    
    return fig
