/* bc_shared_slot.h -- BC_VTP14_N3: one process-wide named slot that kernel libraries share
 * (BigCherry managed patch rad0017).
 *
 * Kernel libraries are dlopened RTLD_LOCAL, so a `static` in one library is invisible to another.
 * A tensor-parallel world whose ranks run different libraries (libr3 on gfx1100, libr4d on gfx1201)
 * still has to meet at one all-reduce rendezvous. The engine exports rad_bc_shared_slot; a library
 * looks it up with dlsym(RTLD_DEFAULT) and gets the same storage as every other library asking for
 * the same key.
 *
 * Fail closed: a key asked for again with a different size or alignment is refused with the reason,
 * never handed a second slot. A slot is constructed once (by the first caller's init) and lives for
 * the process: no destructor is kept, so no callback into an unloaded library remains. */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

extern "C" {
/* RAD_OK and *out = the slot for `key` (constructed by `init` on first use). RAD_E_INVAL for a
 * null or empty key, zero size, an alignment that is not a power of two, or a null out; RAD_E_STATE
 * when the key exists with a different size or alignment (the reason in why[0..why_len)). */
int rad_bc_shared_slot(const char* key, size_t size, size_t align, void (*init)(void* slot), void** out,
                       char* why, size_t why_len);
}

namespace rad {

/* The libraries (their file names, from dladdr of the init or caller address) that resolved `key`,
 * in first-use order: the engine's receipt that ranks of different libraries share one slot. */
std::vector<std::string> bc_shared_slot_users(const std::string& key);

/* The keys that start with `prefix`, sorted. */
std::vector<std::string> bc_shared_slot_keys(const std::string& prefix);

/* Tests: record a user by name, as a library resolving the key would. */
int bc_shared_slot_for(const std::string& user, const char* key, size_t size, size_t align,
                       void (*init)(void*), void** out, std::string* why);

}  /* namespace rad */
