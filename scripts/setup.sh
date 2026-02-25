#!/usr/bin/env bash
set -e

# Change to project root directory (parent of scripts/)
cd "$(dirname "$0")/.." || exit 1

echo "============================================"
echo "  Workshop Setup - Let AI Handle the Mess"
echo "============================================"
echo ""

# Deactivate any active virtual environment first
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Deactivating existing virtual environment..."
    deactivate 2>/dev/null || true
    # Remove venv paths from PATH to avoid finding the venv's Python
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "venv" | tr '\n' ':' | sed 's/:$//')
    echo ""
fi

# Find a compatible Python version (3.12 or 3.13)
PYTHON_CMD=""

for cmd in python3.13 python3.12 python3 python; do
    if command -v "$cmd" &> /dev/null; then
        # Skip if this Python lives inside a venv
        cmd_path=$(command -v "$cmd")
        if [[ "$cmd_path" == *"/venv/"* ]]; then
            continue
        fi
        version=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [[ "$major" == "3" && ("$minor" == "12" || "$minor" == "13") ]]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "Could not find Python 3.12 or 3.13."
    echo "   Python 3.14 is NOT supported by the packages used in this workshop."
    echo ""

    # Attempt to auto-install Python 3.13
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS -- use Homebrew
        if command -v brew &> /dev/null; then
            read -p "   Install Python 3.13 via Homebrew? (Y/n): " confirm
            confirm=${confirm:-Y}
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                echo ""
                echo "Installing Python 3.13 via Homebrew..."
                brew install python@3.13
                PYTHON_CMD=$(brew --prefix python@3.13)/bin/python3.13
                if [[ ! -x "$PYTHON_CMD" ]]; then
                    echo "Installation failed. Please install manually:"
                    echo "https://www.python.org/downloads/"
                    exit 1
                fi
                echo "Python 3.13 installed successfully."
                echo ""
            else
                echo ""
                echo "   Please install Python 3.13 manually from:"
                echo "   https://www.python.org/downloads/"
                exit 1
            fi
        else
            echo "   Homebrew is not installed. You can install it from https://brew.sh"
            echo "   then re-run this script, or install Python 3.13 manually from:"
            echo "   https://www.python.org/downloads/"
            exit 1
        fi
    else
        # Linux -- try apt, dnf, or yum
        if command -v apt-get &> /dev/null; then
            read -p "   Install Python 3.13 via apt? (This may require sudo) (Y/n): " confirm
            confirm=${confirm:-Y}
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                echo ""
                echo "Installing Python 3.13 via apt..."
                sudo apt-get update && sudo apt-get install -y python3.13 python3.13-venv
                PYTHON_CMD="python3.13"
            else
                echo ""
                echo "   Please install Python 3.13 manually from:"
                echo "   https://www.python.org/downloads/"
                exit 1
            fi
        elif command -v dnf &> /dev/null; then
            read -p "   Install Python 3.13 via dnf? (This may require sudo) (Y/n): " confirm
            confirm=${confirm:-Y}
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                echo ""
                echo "Installing Python 3.13 via dnf..."
                sudo dnf install -y python3.13
                PYTHON_CMD="python3.13"
            else
                echo ""
                echo "   Please install Python 3.13 manually from:"
                echo "   https://www.python.org/downloads/"
                exit 1
            fi
        else
            echo "   Could not detect a supported package manager (apt, dnf, brew)."
            echo "   Please install Python 3.13 manually from:"
            echo "   https://www.python.org/downloads/"
            exit 1
        fi
    fi
fi

FULL_VERSION=$("$PYTHON_CMD" --version 2>&1)
echo "  Found compatible Python: $FULL_VERSION ($PYTHON_CMD)"
echo ""

# Create virtual environment
if [[ -d "venv" ]]; then
    echo "  A 'venv' folder already exists."
    read -p "   Do you want to delete it and create a new one? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "   Removed old venv."
    else
        echo "   Keeping existing venv."
    fi
fi

if [[ ! -d "venv" ]]; then
    echo " Creating virtual environment..."
    "$PYTHON_CMD" -m venv venv
    echo " Virtual environment created."
fi

echo ""

# Activate virtual environment
echo " Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo " Installing required packages..."
echo ""
pip install --upgrade pip -q
pip install -r requirements.txt
echo ""

echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To activate the environment in the future, run:"
echo ""
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter Lab, run:"
echo ""
echo "  jupyter lab"
echo ""
