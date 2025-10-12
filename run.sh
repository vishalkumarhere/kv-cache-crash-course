#!/bin/bash

# KV-Cache Playground Launcher

echo "⚡ Starting KV-Cache Playground..."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Launch Streamlit app
streamlit run kv_cache_app.py

echo ""
echo "✅ KV-Cache Playground is running!"
