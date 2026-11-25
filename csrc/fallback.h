#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace iris {

bool is_cpu_fallback_enabled() noexcept;
void set_cpu_fallback_enabled(bool enabled) noexcept;

bool is_fallback_warning_enabled() noexcept;
void set_fallback_warning_enabled(bool enabled) noexcept;

std::unordered_map<std::string, std::uint64_t> fallback_stats();
void reset_fallback_stats();

}  // namespace iris
