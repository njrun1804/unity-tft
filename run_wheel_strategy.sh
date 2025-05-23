#!/bin/bash
#
# run_wheel_strategy.sh - Production deployment script for Unity Trading System
# 
# Usage:
#   ./run_wheel_strategy.sh [mode] [options]
#
# Modes:
#   once       - Single execution cycle (default)
#   continuous - Continuous trading loop
#   backtest   - Run historical backtest
#   train      - Train TFT model
#   test       - Run test suite
#
# Examples:
#   ./run_wheel_strategy.sh once
#   ./run_wheel_strategy.sh continuous
#   ./run_wheel_strategy.sh backtest --start 2023-01-01 --end 2024-01-01
#   ./run_wheel_strategy.sh train --optimize
#

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MODE="${1:-once}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v $PYTHON_BIN &> /dev/null; then
        log_error "Python not found. Please install Python 3.9+ or set PYTHON_BIN environment variable."
        exit 1
    fi
    
    # Check Python version
    python_version=$($PYTHON_BIN -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    if [[ $(echo "$python_version < 3.9" | bc -l) -eq 1 ]]; then
        log_error "Python 3.9+ required. Found: $python_version"
        exit 1
    fi
    
    # Check API key
    if [[ -z "$POLYGON_API_KEY" ]]; then
        log_error "POLYGON_API_KEY environment variable not set."
        log_info "Get your API key from: https://polygon.io/"
        exit 1
    fi
    
    # Check required files
    if [[ ! -f "positions.json" ]]; then
        log_error "positions.json not found. This file contains trading configuration."
        exit 1
    fi
    
    log_info "✅ Prerequisites check passed"
}

# Install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        log_info "Creating virtual environment..."
        $PYTHON_BIN -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install dependencies
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    
    log_info "✅ Dependencies installed"
}

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p data/feature_store/polygon_watchlist
    mkdir -p data/feature_store/polygon_option_chain
    mkdir -p data/factors
    mkdir -p data/positions
    mkdir -p outputs/predictions
    mkdir -p outputs/backtests
    mkdir -p logs
    mkdir -p models/tft
    
    log_info "✅ Directories created"
}

# Run single execution
run_once() {
    log_info "Running single execution cycle..."
    
    export RUN_MODE=once
    export LOG_LEVEL=$LOG_LEVEL
    
    $PYTHON_BIN -m src.pipelines.wheel_orchestrator
    
    log_info "✅ Single execution completed"
}

# Run continuous trading
run_continuous() {
    log_info "Starting continuous trading mode..."
    log_warn "This will run indefinitely. Press Ctrl+C to stop."
    
    export RUN_MODE=continuous
    export LOG_LEVEL=$LOG_LEVEL
    
    # Create systemd-style logging
    $PYTHON_BIN -m src.pipelines.wheel_orchestrator 2>&1 | tee logs/wheel_strategy_$(date +%Y%m%d_%H%M%S).log
}

# Run backtest
run_backtest() {
    log_info "Running backtest..."
    
    shift  # Remove 'backtest' argument
    $PYTHON_BIN -m src.backtesting.backtest_engine "$@"
    
    log_info "✅ Backtest completed"
}

# Train TFT model
train_model() {
    log_info "Training TFT model..."
    
    shift  # Remove 'train' argument
    
    # Default training data paths
    TRAIN_DATA="${TRAIN_DATA:-data/train.csv}"
    VAL_DATA="${VAL_DATA:-data/val.csv}"
    MODEL_DIR="${MODEL_DIR:-models/tft/}"
    
    if [[ ! -f "$TRAIN_DATA" ]]; then
        log_error "Training data not found: $TRAIN_DATA"
        log_info "Please prepare training data or set TRAIN_DATA environment variable"
        exit 1
    fi
    
    $PYTHON_BIN train_tft.py "$TRAIN_DATA" "$VAL_DATA" "$MODEL_DIR" "$@"
    
    log_info "✅ Model training completed"
}

# Run test suite
run_tests() {
    log_info "Running test suite..."
    
    shift  # Remove 'test' argument
    
    # Install test dependencies
    pip install -q pytest pytest-cov pytest-mock
    
    # Run tests
    pytest tests/ -v --cov=src --cov-report=html --cov-report=term "$@"
    
    log_info "✅ Tests completed"
    log_info "Coverage report: htmlcov/index.html"
}

# Legacy midday workflow
run_midday() {
    log_info "Running legacy midday workflow..."
    
    OUTPUT_FILE="${2:-data/factors/u_midday.json}"
    
    $PYTHON_BIN src/midday_ingest.py --out "$OUTPUT_FILE"
    
    log_info "✅ Midday analysis completed: $OUTPUT_FILE"
}

# Market status check
check_market_status() {
    log_info "Checking market status..."
    
    $PYTHON_BIN -c "
from datetime import datetime
from src.pipelines.wheel_orchestrator import is_market_open

now = datetime.now()
if is_market_open(now):
    print('🟢 Market is OPEN')
else:
    print('🔴 Market is CLOSED')
print(f'Current time: {now.strftime(\"%Y-%m-%d %H:%M:%S %Z\")}')
"
}

# Show help
show_help() {
    cat << EOF
Unity Trading System - Production Deployment Script

USAGE:
    ./run_wheel_strategy.sh [MODE] [OPTIONS]

MODES:
    once          Run single execution cycle (default)
    continuous    Run continuous trading loop  
    backtest      Run historical backtest
    train         Train TFT model
    test          Run test suite
    midday        Legacy midday workflow
    status        Check market status
    help          Show this help

ENVIRONMENT VARIABLES:
    POLYGON_API_KEY    Required: Your Polygon.io API key
    LOG_LEVEL         Optional: DEBUG|INFO|WARNING|ERROR (default: INFO)
    PYTHON_BIN        Optional: Python executable (default: python)
    RUN_MODE          Auto-set based on mode

EXAMPLES:
    # Single execution
    ./run_wheel_strategy.sh once
    
    # Continuous trading
    ./run_wheel_strategy.sh continuous
    
    # Backtest with date range  
    ./run_wheel_strategy.sh backtest --start 2023-01-01 --end 2024-01-01
    
    # Train with optimization
    ./run_wheel_strategy.sh train --optimize
    
    # Legacy workflow
    ./run_wheel_strategy.sh midday

SAFETY:
    - Start with paper trading to validate signals
    - Monitor positions closely during initial deployment
    - Keep position sizes small until strategy is proven
    - Use kill switch (Ctrl+C) to stop continuous mode

For more information, see README.md
EOF
}

# Main execution
main() {
    case "$MODE" in
        "once")
            check_prerequisites
            install_dependencies
            setup_directories
            run_once
            ;;
        "continuous")
            check_prerequisites
            install_dependencies
            setup_directories
            run_continuous
            ;;
        "backtest")
            check_prerequisites
            install_dependencies
            run_backtest "$@"
            ;;
        "train")
            check_prerequisites
            install_dependencies
            setup_directories
            train_model "$@"
            ;;
        "test")
            check_prerequisites
            install_dependencies
            run_tests "$@"
            ;;
        "midday")
            check_prerequisites
            install_dependencies
            setup_directories
            run_midday "$@"
            ;;
        "status")
            check_market_status
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            log_error "Unknown mode: $MODE"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
