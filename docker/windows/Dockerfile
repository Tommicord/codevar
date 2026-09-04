# Copyright 2026 Codevar
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Dockerfile for Windows builds
FROM --platform=linux/amd64 rust:1.93.0-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    clang \
    libclang-dev \
    mingw-w64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./
COPY codevar-core ./codevar-core
COPY codevar-colab ./codevar-colab
COPY codevar-fparser ./codevar-fparser

# Add Windows targets
RUN rustup target add x86_64-pc-windows-gnu

# Build for Windows
RUN cargo build --workspace --release --target x86_64-pc-windows-gnu

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built artifacts from builder
COPY --from=builder /app/target/x86_64-pc-windows-gnu/release/codevar_core.dll /usr/local/lib/
COPY --from=builder /app/target/x86_64-pc-windows-gnu/release/codevar_colab.dll /usr/local/lib/
COPY --from=builder /app/target/x86_64-pc-windows-gnu/release/codevar_fparser.dll /usr/local/lib/

CMD ["bash"]