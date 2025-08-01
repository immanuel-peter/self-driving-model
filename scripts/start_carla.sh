#!/bin/bash

echo "🚀 Starting CARLA for Tesla FSD preparation..."

# 1. Set up storage directories (default: $HOME/carla_data unless overridden)
CARLA_DATA_PATH="${CARLA_DATA_PATH:-$HOME/carla_data}"
echo "📂 Using CARLA data directory: $CARLA_DATA_PATH"

sudo mkdir -p "$CARLA_DATA_PATH/automoe_training"

export CARLA_DATA_PATH

# 2. Start virtual display
export DISPLAY=:1
echo "📺 Starting virtual display..."
if ! pgrep Xvfb > /dev/null; then
    Xvfb :1 -screen 0 1024x768x24 &
    sleep 2
fi

# 3. Launch CARLA server
echo "🎮 Starting CARLA server..."
CARLA_DIR="${CARLA_DIR:-/opt/carla-simulator}"  # default, override if needed
if [ ! -d "$CARLA_DIR" ]; then
    echo "❌ CARLA directory not found: $CARLA_DIR"
    exit 1
fi
cd "$CARLA_DIR"
DISPLAY=:1 ./CarlaUE4.sh -prefernvidia -RenderOffScreen -opengl -carla-rpc-port=2000 &

echo "⏳ CARLA is initializing (this takes 2-3 minutes)..."
echo "✅ Run 'python3 test_carla.py' in another terminal to verify connection"

