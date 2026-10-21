// Read a response file in place of Chromium's HTTPS request.
#include "pkresolve.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string_view>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: resolve <pubky-service> <bundle-json-file>\n";
    return 1;
  }
  auto client = pkresolve::new_client();
  const std::vector protocols = {pkresolve::HttpProtocol::Http2,
                                 pkresolve::HttpProtocol::Http11};
  auto started = std::chrono::steady_clock::now();
  auto preparation = client->prepare(
      argv[1], 443, {protocols.data(), protocols.size()});
  pkresolve::ResolveResponse response{};
  std::size_t bundle_reads = 0;
  if (preparation->needs_bundle()) {
    const auto request_body = preparation->request_body();
    std::cout << "POST /v1/resolve: "
              << std::string_view(request_body.data(), request_body.size())
              << '\n';
    // Chromium retains preparation while its network request is in flight.
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) {
      std::cerr << "could not open response bundle\n";
      return 1;
    }
    const std::vector<std::uint8_t> body(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    ++bundle_reads;
    started = std::chrono::steady_clock::now();
    response = client->accept_bundle(std::move(preparation),
                                    {body.data(), body.size()});
  } else {
    response = pkresolve::prepared_result(std::move(preparation));
  }
  if (response.status.code != pkresolve::StatusCode::Ok) {
    std::cerr << std::string_view(response.status.message.data(),
                                  response.status.message.size())
              << '\n';
    return 1;
  }
  // Preserve this deadline when copying the result or crossing IPC.
  const auto expires_at =
      started + std::chrono::milliseconds(response.resolution.valid_for_ms);
  if (expires_at <= std::chrono::steady_clock::now()) {
    std::cerr << "resolution expired\n";
    return 1;
  }
  const auto& resolution = response.resolution;
  std::cout << "SNI: "
            << std::string_view(resolution.service.data(),
                                resolution.service.size())
            << '\n';
  for (const auto& candidate : resolution.candidates) {
    for (const auto& address : candidate.addresses) {
      std::cout << std::string_view(address.data(), address.size()) << ':'
                << candidate.port << '\n';
    }
  }
  auto cached = client->prepare(
      argv[1], 443, {protocols.data(), protocols.size()});
  if (cached->needs_bundle() ||
      pkresolve::prepared_result(std::move(cached)).status.code !=
          pkresolve::StatusCode::Ok) {
    std::cerr << "cached resolution failed\n";
    return 1;
  }
  std::cout << "Bundle reads after two resolutions: " << bundle_reads << '\n';
  // Verify possession of resolution.public_key when opening destination TLS.
  return 0;
}
