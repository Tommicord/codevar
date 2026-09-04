FROM rust:1.93-bookworm as builder
WORKDIR /usr/src/codevar

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY codevar-core/Cargo.toml core/Cargo.toml

RUN mkdir -p codevar-core/src
RUN echo "fn main() { }" > codevar-core/src/lib.rs

RUN cargo fetch

COPY . ./

RUN cargo build --workspace --release

FROM debian:bookworm-slim
WORKDIR /usr/src/codevar

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/codevar/target/release /usr/src/codevar/target/release

CMD ["bash", "-lc", "ls -1 target/release"]
