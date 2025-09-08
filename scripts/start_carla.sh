#!/bin/bash

echo "🚀 Starting CARLA for Tesla FSD preparation..."

CARLA_DATA_PATH="${CARLA_DATA_PATH:-datasets/carla/raw}"
echo "📂 Using CARLA data directory: $CARLA_DATA_PATH"

mkdir -p "$CARLA_DATA_PATH/raw"

export CARLA_DATA_PATH

# Start virtual display
export DISPLAY=:1
echo "📺 Starting virtual display..."
if ! pgrep Xvfb > /dev/null; then
    Xvfb :1 -screen 0 1024x768x24 &
    sleep 2
fi

# Launch CARLA server
echo "🎮 Starting CARLA server..."
CARLA_DIR="${CARLA_DIR:-/path/to/carla-simulator}"
if [ ! -d "$CARLA_DIR" ]; then
    echo "❌ CARLA directory not found: $CARLA_DIR"
    exit 1
fi
cd "$CARLA_DIR"
DISPLAY=:1 ./CarlaUnreal.sh -prefernvidia -RenderOffScreen -opengl -carla-rpc-port=2000 &
CARLA_PID=$!
sleep 5

if ! ps -p $CARLA_PID > /dev/null; then
    echo "❌ CARLA server failed to start. Check logs above for errors."
    exit 2
else
    echo "⏳ CARLA is initializing (this takes 2-3 minutes)..."
    echo "✅ Run 'python3 test_carla.py' in another terminal to verify connection"
fi