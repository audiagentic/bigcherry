Engine::~Engine() {
    for (auto& r : ranks_) {
        if (!r->heat.enabled()) continue;
        RAD_INFO("%s", r->mover.report().c_str());
    }
    for (auto& r : ranks_) r->ctx.shutdown();
    step_state_fini();
}
