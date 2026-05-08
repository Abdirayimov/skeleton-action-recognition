#pragma once

#include <spdlog/spdlog.h>

#include <string>

namespace skeleton_ar::utils {

/// Initialise the global spdlog logger.
void init_logger(const std::string& level, bool json);

}  // namespace skeleton_ar::utils

#define SAR_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define SAR_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define SAR_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define SAR_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define SAR_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define SAR_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
