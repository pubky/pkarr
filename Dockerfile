# ========================
# Build Stage
# ========================
FROM rust:alpine3.23 AS builder

# Build platform argument

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    build-base \
    curl

# Set the working directory
WORKDIR /usr/src/app

# Copy over Cargo.toml and Cargo.lock for dependency caching
COPY Cargo.toml Cargo.lock ./

# Copy over all the source code
COPY . .

# Build the relay in release mode
RUN cargo build -p pkarr-relay --release

# Strip the binary to reduce size
RUN strip target/release/pkarr-relay

# ========================
# Runtime Stage
# ========================
FROM alpine:3.20

# Install runtime dependencies (only ca-certificates)
RUN apk add --no-cache ca-certificates

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/target/release/pkarr-relay /usr/local/bin/pkarr-relay

# Set the working directory
WORKDIR /usr/local/bin

# Expose the port the pkarr relay listens on (should match that of config.toml)
EXPOSE 6881

# Set the default command to run the relay binary
CMD ["pkarr-relay", "--config=/config.toml"]
