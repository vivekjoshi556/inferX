#ifndef INFERX_UTILS_H
#define INFERX_UTILS_H

#include <stdint.h>
#include "inferX/types.h"

namespace inferX {
Device get_device(const std::string&);

/**
 * Returns Available Memory for given device in bytes.
 */
size_t get_memory_available(const Device&);

size_t cpu_available_memory(const Device&);
size_t cuda_available_memory(const Device&);

/**
 * Returns given mem bytes in human readble form
 */
std::string human_readable_memory(const uint64_t&);
}  // namespace inferX

#endif