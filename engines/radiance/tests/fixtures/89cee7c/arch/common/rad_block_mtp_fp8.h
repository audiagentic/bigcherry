inline bool mtp_draft_head_q2(RadBuilder* b, const char* head, const char* lm) {
    if (!head || !lm) return false;
    if (map_copy(b, head, lm) < 0) return false;
    RadEncoding e{};
    return weight_enc(b, head, &e) && e.n_planes > 0 && e.plane[0].dtype == RAD_U2;
}
