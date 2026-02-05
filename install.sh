#!/bin/bash
# gt-audit installer
# Usage: curl -fsSL https://raw.githubusercontent.com/ARTIFACTIQ/gt-audit/main/install.sh | bash

set -e

REPO="ARTIFACTIQ/gt-audit"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    linux)
        case "$ARCH" in
            x86_64) PLATFORM="linux-x86_64" ;;
            *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    darwin)
        case "$ARCH" in
            arm64) PLATFORM="darwin-arm64" ;;
            x86_64)
                echo "Note: macOS x86_64 requires Rosetta 2 to run ARM64 binary"
                PLATFORM="darwin-arm64"
                ;;
            *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

echo "Installing gt-audit for $PLATFORM..."

# Get latest release
LATEST=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | cut -d'"' -f4)
echo "Latest version: $LATEST"

# Download
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$LATEST/gt-audit-$PLATFORM.tar.gz"
echo "Downloading from: $DOWNLOAD_URL"

TEMP_DIR=$(mktemp -d)
curl -fsSL "$DOWNLOAD_URL" -o "$TEMP_DIR/gt-audit.tar.gz"

# Extract
tar -xzf "$TEMP_DIR/gt-audit.tar.gz" -C "$TEMP_DIR"

# Install
mkdir -p "$INSTALL_DIR"
mv "$TEMP_DIR/gt-audit" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/gt-audit"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "gt-audit installed to: $INSTALL_DIR/gt-audit"
echo ""

# Check if in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "Add to your PATH:"
    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    echo ""
fi

echo "Usage:"
echo "  gt-audit validate /path/to/dataset --model model.onnx"
echo "  gt-audit info /path/to/dataset"
echo ""
echo "Documentation: https://github.com/$REPO"
