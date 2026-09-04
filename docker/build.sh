#!/bin/bash
# Copyright 2026 Codevar
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <platform>"
    echo "Platforms:"
    echo "  linux    - Build for Linux (default)"
    echo "  macos    - Build for macOS (cross-compilation)"
    echo "  windows  - Build for Windows (cross-compilation)"
    echo "  wasm     - Build for WebAssembly"
    echo "  all      - Build for all platforms"
    echo "  dev      - Start development container"
    exit 1
}
if [ $# -eq 0 ]; then
    PLATFORM="linux"
else
    PLATFORM=$1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

build_linux() {
    print_status "Building for Linux"
    docker build -f docker/Dockerfile.linux -t codevar:linux-latest .
    print_status "Build complete"
}

build_macos() {
    print_status "Building for macOS (cross-compilation)"
    docker build -f docker/Dockerfile.macos -t codevar:macos-latest .
    print_status "macOS build complete"
}

build_windows() {
    print_status "Building for Windows (cross-compilation)"
    docker build -f docker/Dockerfile.windows -t codevar:windows-latest .
    print_status "Build complete"
}

build_wasm() {
    print_status "Building for WebAssembly"
    docker build -f docker/Dockerfile.wasm -t codevar:wasm-latest .
    print_status "Build complete"
}

build_all() {
    print_status "Building for all platforms"
    build_linux
    build_macos
    build_windows
    build_wasm
    print_status "All platforms build complete"
}

start_dev() {
    print_status "Starting development container"
    docker-compose -f docker/docker-compose.yml run --rm dev
}

case $PLATFORM in
    linux)
        build_linux
        ;;
    macos)
        build_macos
        ;;
    windows)
        build_windows
        ;;
    wasm)
        build_wasm
        ;;
    all)
        build_all
        ;;
    dev)
        start_dev
        ;;
    *)
        print_error "Unknown platform: $PLATFORM"
        show_usage
        ;;
esac