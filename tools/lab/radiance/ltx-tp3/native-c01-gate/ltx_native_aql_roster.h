// Shared only by the LTX native C05/C06 diagnostic tests.
// Device mapping/admission for production belongs to RUM05; these
// checks do not open a native Engine or launch an alternative collective.
#pragma once
#include "device/device.h"
#include "plugin/registry.h"

#include <cstring>
#include <stdexcept>

namespace ltx_test {
inline void require_aql_backend() {
    if (rad::device_is_host() ||
        std::strcmp(rad::device_backend_name(), "aql") != 0)
        throw std::runtime_error("LTX diagnostic refuses host/direct HIP backend in AQL mode");
    if (rad_dev_count() != 3)
        throw std::runtime_error("LTX diagnostic AQL needs exactly three visible devices");
}

inline void configure_aql_registry(rad::Registry& reg) {
    if (reg.set_code_devices({0,1,2}) != RAD_OK)
        throw std::runtime_error("LTX diagnostic AQL code-device roster refused");
    int64_t p2p = 1;
    for (int rank=0;rank<3;++rank) {
        RadDeviceProps props{};
        if (rad_dev_props(rank,&props) != RAD_OK ||
            (std::strcmp(props.arch,"gfx1100") != 0 &&
             std::strcmp(props.arch,"gfx1201") != 0))
            throw std::runtime_error("LTX diagnostic AQL device architecture unsupported");
        for (int peer=0;peer<3;++peer) if (peer != rank) {
            const int cap = rad_dev_peer_access(rank,peer);
            if (cap < 0)
                throw std::runtime_error("LTX diagnostic AQL peer-access query failed");
            if (!cap) p2p=0;
        }
    }
    reg.set_device_fact(RAD_FACT_P2P,p2p);
}
}  // namespace ltx_test
