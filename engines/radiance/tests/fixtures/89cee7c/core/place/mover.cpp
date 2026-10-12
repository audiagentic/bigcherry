void Residency::publish(rad_weight w, void* ptr) {
    if ((size_t)w >= slot_.size()) return;
    slot_[w].ptr = ptr;
    slot_[w].ready = nullptr;
    ++slot_[w].generation;
    ++epoch_;
    note(w);
}

void Residency::begin_transfer(rad_weight w, void* ptr, RadEvent ready) {
    if ((size_t)w >= slot_.size()) return;
    slot_[w].ptr = ptr;
    slot_[w].ready = ready;
    ++slot_[w].generation;
    ++epoch_;
    note(w);
}

void Mover::republish(int32_t unit) {
    const MoveUnit& mu = plan_->units[(size_t)unit];
    char* base = (char*)unit_base(unit);
    for (size_t i = 0; i < mu.weights.size(); ++i)
        res_.publish(mu.weights[i], base ? base + mu.offset[i] : nullptr);
}

int Mover::stage_units() {
    for (size_t k = 0; k < staged->size(); ++k) {
        const MoveUnit& mu = plan_->units[(size_t)(*staged)[k]];
        for (size_t i = 0; i < mu.weights.size(); ++i)
            res_.publish(mu.weights[i], dst + at[k] + mu.offset[i]);
    }
    return RAD_OK;
}
