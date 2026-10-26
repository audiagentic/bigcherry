#pragma once
// Included only by r3_moe_worker.cpp, after worker_container.h (stopping()).
//
// RAD20 queue warm-up (power save on). On gfx11, a GPU whose active compute queues exceed its hardware slots keeps MES
// oversubscribed and busy; that costs ~85 W at idle but dispatches the helper's kernel chains faster (decode launch gap
// p95 3.6 vs 7.6 us, rad20-queue-probe.sh). So during an inference burst the helper holds just enough extra idle queues
// to oversubscribe the GPU, and parks them between bursts: KFD counts a queue as active only while its percentage is
// above 0 (QUEUE_IS_ACTIVE), and AMDKFD_IOC_UPDATE_QUEUE with 0 removes it from MES, 100 adds it back -- what ROCr's
// own AqlQueue::Suspend/Resume do. The placeholders are public HSA queues that never receive packets; this process's
// /dev/kfd handle reaches them because KFD keys queues by process. Switching costs ~150-250 us a queue
// (rad20-queue-warm.cpp), done on this class's own thread so the engine never waits for it.
//
// The count is derived, not configured: the GPU's compute slots (KFD topology num_cp_queues) minus the queues this
// process already owns on it (/sys/class/kfd/kfd/proc/<pid>/queues), plus one. R3_MOE_WORKER_QUEUE_WARM=0 turns it off.
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

class QueueWarm {
public:
    QueueWarm() = default;
    QueueWarm(const QueueWarm&) = delete;
    QueueWarm& operator=(const QueueWarm&) = delete;
    ~QueueWarm() { stop(); }

    /* Create the placeholders (parked) and start the thread that follows the engine's bursts. False only on an error
     * that leaves the helper unable to run, with everything already created released; a disabled or unneeded warm-up
     * returns true with no queues. */
    bool start(r3::moe_shm::Header* h, bool enabled) {
        if (!enabled) return true;
        if (init(h)) return true;
        stop();
        return false;
    }

    /* After the helper's stop path has set kBurstStop and woken Header::burst. Safe on a partial start. */
    void stop() {
        if (thread_.joinable()) thread_.join();
        if (kfd_ >= 0 && !queues_.empty()) (void)set_all(100);   /* hand ROCr back queues in the state it made them */
        for (auto* q : queues_) if (q) (void)hsa_queue_destroy(q);
        queues_.clear();
        if (kfd_ >= 0) close(kfd_);
        kfd_ = -1;
        if (hsa_open_) (void)hsa_shut_down();   /* reference counted: HIP's own use of HSA is unaffected */
        hsa_open_ = false;
    }

private:
    bool init(r3::moe_shm::Header* h) {
        if (hsa_init() != HSA_STATUS_SUCCESS) return fail("hsa_init");
        hsa_open_ = true;
        hsa_agent_t gpu{};
        const hsa_status_t it = hsa_iterate_agents(first_gpu, &gpu);
        if ((it != HSA_STATUS_SUCCESS && it != HSA_STATUS_INFO_BREAK) || !gpu.handle) return fail("no GPU agent");
        uint32_t gpu_id = 0;
        if (hsa_agent_get_info(gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_UID), &gpu_id) !=
            HSA_STATUS_SUCCESS)
            return fail("agent gpu_id");
        const int slots = topology_cp_queues(gpu_id);
        const int owned = own_queues(gpu_id);
        if (slots <= 0 || owned < 0) return fail("KFD topology / process queues");
        const int need = slots + 1 - owned;
        std::fprintf(stderr, "r3_moe_worker: queue warm-up: %d compute slots, %d queues owned, %d placeholder(s)\n",
                     slots, owned, need > 0 ? need : 0);
        if (need <= 0) return true;   /* already oversubscribed at all times (e.g. power save off) */
        kfd_ = open("/dev/kfd", O_RDWR | O_CLOEXEC);
        if (kfd_ < 0) return fail("open /dev/kfd");
        for (int i = 0; i < need; ++i) {
            hsa_queue_t* q = nullptr;
            if (hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q) !=
                HSA_STATUS_SUCCESS)
                return fail("hsa_queue_create");
            queues_.push_back(q);
        }
        if (!set_all(0)) return false;
        h_ = h;
        try {
            thread_ = std::thread([this] { follow(); });
        } catch (const std::exception&) {
            return fail("thread");
        }
        return true;
    }

    static hsa_status_t first_gpu(hsa_agent_t a, void* out) {
        hsa_device_type_t t{};
        if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) == HSA_STATUS_SUCCESS && t == HSA_DEVICE_TYPE_GPU) {
            *static_cast<hsa_agent_t*>(out) = a;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    }
    static bool fail(const char* what) {
        std::fprintf(stderr, "r3_moe_worker: queue warm-up: %s failed (%s)\n", what, std::strerror(errno));
        return false;
    }
    static int read_int(const std::string& path, const char* key) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return -1;
        char k[64];
        long long v = -1, out = -1;
        while (std::fscanf(f, "%63s %lld", k, &v) == 2)
            if (std::strcmp(k, key) == 0) out = v;
        std::fclose(f);
        return (int)out;
    }
    static int topology_cp_queues(uint32_t gpu_id) {
        const std::string base = "/sys/class/kfd/kfd/topology/nodes/";
        DIR* d = opendir(base.c_str());
        if (!d) return -1;
        int slots = -1;
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            FILE* f = std::fopen((base + e->d_name + "/gpu_id").c_str(), "r");
            unsigned id = 0;
            const bool match = f && std::fscanf(f, "%u", &id) == 1 && id == gpu_id;
            if (f) std::fclose(f);
            if (match) slots = read_int(base + e->d_name + "/properties", "num_cp_queues");
        }
        closedir(d);
        return slots;
    }
    static int own_queues(uint32_t gpu_id) {
        const std::string base = "/sys/class/kfd/kfd/proc/" + std::to_string(getpid()) + "/queues/";
        DIR* d = opendir(base.c_str());
        if (!d) return -1;
        int n = 0;
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            FILE* f = std::fopen((base + e->d_name + "/gpuid").c_str(), "r");
            unsigned id = 0;
            if (f && std::fscanf(f, "%u", &id) == 1 && id == gpu_id) ++n;
            if (f) std::fclose(f);
        }
        closedir(d);
        return n;
    }
    bool set_all(uint32_t pct) {
        for (auto* q : queues_) {
            kfd_ioctl_update_queue_args a{};
            a.ring_base_address = reinterpret_cast<uint64_t>(q->base_address);
            a.queue_id = static_cast<uint32_t>(q->id);
            a.ring_size = q->size * 64u;
            a.queue_percentage = pct;
            a.queue_priority = 7;   /* libhsakmt priority_map[HSA_QUEUE_PRIORITY_NORMAL + 3] */
            if (ioctl(kfd_, AMDKFD_IOC_UPDATE_QUEUE, &a) != 0) return fail("AMDKFD_IOC_UPDATE_QUEUE");
        }
        return true;
    }
    /* Active while the engine is inside a burst, parked otherwise. Sleeps on the burst futex between transitions. */
    void follow() {
        using namespace r3::moe_shm;
        bool active = false;
        for (;;) {
            const uint32_t b = load(&h_->burst);
            if ((b & kBurstStop) || stopping()) break;
            const bool want = b & 1u;
            if (want != active) {
                if (!set_all(want ? 100 : 0)) { request_stop(); break; }
                active = want;
                continue;   /* the burst may have moved while switching */
            }
            (void)futex_wait(&h_->burst, b);
        }
    }

    r3::moe_shm::Header* h_ = nullptr;
    std::vector<hsa_queue_t*> queues_;
    int kfd_ = -1;
    bool hsa_open_ = false;
    std::thread thread_;
};
