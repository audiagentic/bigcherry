/* bc_shared_slot.cpp -- BC_VTP14_N3: see bc_shared_slot.h. */
#include "bc_shared_slot.h"

#include "rad_abi.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <new>

namespace rad {

namespace {

struct Slot {
    void*  p     = nullptr;
    size_t size  = 0;
    size_t align = 0;
    std::vector<std::string> users;
};

std::mutex& slots_mu() {
    static std::mutex m;
    return m;
}

/* Never destroyed: the slots outlive every library that uses them, and a rank thread may still be
 * inside one at exit. */
std::map<std::string, Slot>& slots() {
    static auto* s = new std::map<std::string, Slot>();
    return *s;
}

std::string user_of(const void* addr) {
    Dl_info info{};
    if (addr && dladdr(addr, &info) && info.dli_fname) {
        const char* base = std::strrchr(info.dli_fname, '/');
        return base ? base + 1 : info.dli_fname;
    }
    return "?";
}

}  /* namespace */

int bc_shared_slot_for(const std::string& user, const char* key, size_t size, size_t align,
                       void (*init)(void*), void** out, std::string* why) {
    if (!key || !*key || !size || !out || align == 0 || (align & (align - 1)) != 0) {
        if (why) *why = "a shared slot needs a key, a size, a power-of-two alignment and somewhere to put it";
        return RAD_E_INVAL;
    }
    *out = nullptr;
    std::lock_guard<std::mutex> lk(slots_mu());
    auto it = slots().find(key);
    if (it == slots().end()) {
        void* p = ::operator new(size, std::align_val_t(align), std::nothrow);
        if (!p) {
            if (why) *why = "out of memory for shared slot '" + std::string(key) + "'";
            return RAD_E_NOMEM;
        }
        std::memset(p, 0, size);
        if (init) init(p);
        it = slots().emplace(key, Slot{ p, size, align, {} }).first;
    } else if (it->second.size != size || it->second.align != align) {
        if (why)
            *why = "shared slot '" + std::string(key) + "' is " + std::to_string(it->second.size) + " bytes aligned " +
                   std::to_string(it->second.align) + " for " +
                   (it->second.users.empty() ? std::string("?") : it->second.users.front()) + ", and " + user +
                   " asks for " + std::to_string(size) + " aligned " + std::to_string(align) +
                   "; the libraries disagree on its layout, so they cannot share it";
        return RAD_E_STATE;
    }
    if (std::find(it->second.users.begin(), it->second.users.end(), user) == it->second.users.end())
        it->second.users.push_back(user);
    *out = it->second.p;
    return RAD_OK;
}

std::vector<std::string> bc_shared_slot_users(const std::string& key) {
    std::lock_guard<std::mutex> lk(slots_mu());
    auto it = slots().find(key);
    return it == slots().end() ? std::vector<std::string>() : it->second.users;
}

std::vector<std::string> bc_shared_slot_keys(const std::string& prefix) {
    std::lock_guard<std::mutex> lk(slots_mu());
    std::vector<std::string> out;
    for (const auto& kv : slots())
        if (kv.first.compare(0, prefix.size(), prefix) == 0) out.push_back(kv.first);
    return out;
}

}  /* namespace rad */

extern "C" int rad_bc_shared_slot(const char* key, size_t size, size_t align, void (*init)(void*), void** out,
                                  char* why, size_t why_len) {
    std::string w;
    const int rc = rad::bc_shared_slot_for(rad::user_of(init ? (const void*)init : __builtin_return_address(0)),
                                           key, size, align, init, out, &w);
    if (rc != RAD_OK && why && why_len) std::snprintf(why, why_len, "%s", w.c_str());
    return rc;
}
