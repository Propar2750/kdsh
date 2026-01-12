#!/bin/bash
# Quick setup verification script for hackathon developers

echo "=========================================="
echo "KDSH Track A - Setup Verification"
echo "=========================================="
echo ""

# Check Docker
echo "[1/5] Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✓ Docker found: $(docker --version)"
else
    echo "✗ Docker not found! Please install Docker."
    exit 1
fi

# Check Docker Compose
echo ""
echo "[2/5] Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo "✓ Docker Compose found: $(docker-compose --version)"
elif docker compose version &> /dev/null; then
    echo "✓ Docker Compose found (plugin): $(docker compose version)"
else
    echo "✗ Docker Compose not found! Please install Docker Compose."
    exit 1
fi

# Check GROQ_API_KEY
echo ""
echo "[3/5] Checking GROQ_API_KEY..."
if [ -z "$GROQ_API_KEY" ]; then
    echo "✗ GROQ_API_KEY not set!"
    echo "   Get your free API key from: https://console.groq.com/keys"
    echo "   Then run: export GROQ_API_KEY='your-key-here'"
    exit 1
else
    echo "✓ GROQ_API_KEY is set"
fi

# Check Dataset
echo ""
echo "[4/5] Checking Dataset..."
if [ -d "Dataset" ]; then
    echo "✓ Dataset directory found"
    if [ -f "Dataset/train.csv" ]; then
        echo "  ✓ train.csv found"
    else
        echo "  ✗ train.csv not found!"
        exit 1
    fi
    if [ -d "Dataset/Books" ]; then
        book_count=$(ls -1 Dataset/Books/*.txt 2>/dev/null | wc -l)
        echo "  ✓ Books directory found ($book_count books)"
    else
        echo "  ✗ Books directory not found!"
        exit 1
    fi
else
    echo "✗ Dataset directory not found!"
    exit 1
fi

# Build Docker image
echo ""
echo "[5/5] Building Docker image (this may take a few minutes)..."
if docker-compose build; then
    echo "✓ Docker image built successfully"
else
    echo "✗ Docker build failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup verification complete!"
echo "=========================================="
echo ""
echo "To run a quick test:"
echo "  docker-compose run --rm pipeline python -m pipeline.run_eval_fast --max-samples 5"
echo ""
echo "To generate submission file:"
echo "  docker-compose run --rm pipeline python -m pipeline.run_eval_fast --test --out results.csv"
echo ""
echo "For interactive shell:"
echo "  docker-compose run --rm pipeline bash"
echo ""
