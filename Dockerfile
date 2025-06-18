# ========================
# Build Stage
# ========================
FROM rust:1.86.0-alpine3.20 AS builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    build-base \
    curl

# Add the MUSL target for static linking
RUN rustup target add x86_64-unknown-linux-musl

# Set the working directory
WORKDIR /usr/src/app

# Copy over Cargo.toml and Cargo.lock for dependency caching
COPY Cargo.toml Cargo.lock ./

# Copy over all the source code
COPY . .

# Build the project in release mode for the MUSL target
RUN cargo build --release --target x86_64-unknown-linux-musl

# Strip the binary to reduce size
RUN strip target/x86_64-unknown-linux-musl/release/pkarr-relay

# ========================
# Runtime Stage
# ========================
FROM alpine:3.20

# Install runtime dependencies (only ca-certificates)
RUN apk add --no-cache ca-certificates

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/target/x86_64-unknown-linux-musl/release/pkarr-relay /usr/local/bin/pkarr-relay

# Set the working directory
WORKDIR /usr/local/bin

# Expose the port the pkarr relay listens on (should match that of config.toml)
EXPOSE 6881

# Set the default command to run the relay binary
CMD ["pkarr-relay", "--config=./config.toml"]
