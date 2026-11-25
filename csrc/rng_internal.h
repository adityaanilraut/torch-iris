#pragma once

#include <ATen/core/Generator.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace iris {

std::pair<std::uint64_t, std::uint64_t> reserve_rng_blocks_opt(
    std::uint64_t blocks, const std::optional<at::Generator>& generator);

}  // namespace iris
