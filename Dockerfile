# Start with an official Rust image
FROM rust:latest as builder

# Create a new directory for the project
WORKDIR /usr/src/pkarr-server

# Copy the project files into the container
COPY . .

# Build the project in release mode
RUN cargo build --release

# Use an image with a compatible glibc version
FROM ubuntu:latest

# Install necessary libraries
RUN apt-get update && apt-get install -y libssl-dev

# Copy the binary from the builder stage
COPY --from=builder /usr/src/pkarr-server/target/release/pkarr-server /usr/local/bin/pkarr-server

# Copy the example configuration file as config.toml if config.toml is not present
COPY --from=builder /usr/src/pkarr-server/server/src/config.example.toml /etc/pkarr/config.toml

# Expose the necessary port (change according to your server settings)
EXPOSE 6881

# Run the server with the appropriate flags
CMD ["pkarr-server", "--config=/etc/pkarr/config.toml", "-t=pkarr=debug,tower_http=debug"]