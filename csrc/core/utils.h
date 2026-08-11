#ifndef INFERX_UTILS_H
#define INFERX_UTILS_H

#include <stdint.h>
#include "inferX/types.h"

namespace inferX {
Device get_device(const std::string&);

/**
 * Returns given mem bytes in human readble form
 */
std::string human_readable_memory(const uint64_t&);
}  // namespace inferX

#endif