/* bc_serving_managed.h -- the managed native entry (RSC02): the engine starts from the stock `--config FILE` YAML entry
 * and the stock Fleet, and the owner adds only what it derived: `--bc-admission-fd FD` names one sealed document
 * (v2/engine_admission.h). Without that flag nothing in this file changes anything: the stock entry is untouched.
 *
 * The three hooks, in start-up order:
 *   bc_serving_managed_take_flag  before rad0011 takes its flags: strips the owner's flags, installs the admitted state
 *                                 the plan check, the typed engine settings and rad0002/rad0055 read.
 *   bc_serving_managed_bind       after the YAML is parsed and before Fleet::run: the single-model managed contract and
 *                                 the derived values bound onto the parsed Config (absent/auto or exactly agreeing).
 *   bc_serving_managed_chosen     in Fleet::load_op before the Engine exists: the cards Fleet chose are the admitted
 *                                 ranks, in order (rad0016).
 * bc_serving_managed() is true from a successful bind to process exit. It is process-global on purpose and never cleared:
 * a managed process must not become a stock one, and the guards that read it (reload, SIGHUP, admin mutation, restart
 * after an engine exit) run on other threads. */
#ifndef BC_SERVING_MANAGED_H
#define BC_SERVING_MANAGED_H

#include <string>
#include <vector>

namespace rad::fleet { struct FleetConfig; }
namespace bc::serving { struct EngineAdmission; }

/* 0: nothing to do, or done. Otherwise the exit code (2), after one message on stderr. */
int bc_serving_managed_take_flag(int* argc, char*** argv);
int bc_serving_managed_bind(rad::fleet::FleetConfig* fleet);
bool bc_serving_managed();
/* The last refusal message of the hooks above (also printed on stderr), for tests. */
const std::string& bc_serving_managed_last_refusal();
/* Fleet::reap: a managed engine stopped while serving. The process ends (no demand-driven restart) and exits non-zero. */
void bc_serving_managed_note_engine_exit();
bool bc_serving_managed_engine_exited();
/* Empty when `chosen` is the admitted rank order, else the refusal naming both. */
std::string bc_serving_managed_chosen(const std::vector<int>& chosen);

/* Between the two halves of this entry (the flag step and the bind): the one refusal path, the one place the process becomes
 * managed, and the admitted document. */
int bc_serving_managed_refuse(const std::string& why);   /* records, prints, returns 2 */
void bc_serving_managed_mark();
const bc::serving::EngineAdmission* bc_serving_managed_admission();
int bc_serving_managed_admission_fd();   /* the inherited descriptor the admission came from */

/* Defined beside the getters it fills (bc_serving_manifest.cpp). */
void bc_serving_install_engine_state(const bc::serving::EngineAdmission& admission, const std::string& heat_nonce,
                                     const std::string& step_phase_dir);

#endif /* BC_SERVING_MANAGED_H */
