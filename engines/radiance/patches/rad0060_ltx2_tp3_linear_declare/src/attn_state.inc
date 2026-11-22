    /* BC_LTX3_C06_ATTN_STATE: unselected TP3 self-attention uses the
     * same Attn owner, kernel registry and step order as LTX TP1. */
    Tp3GlobalRms tp_q{}, tp_k{};
    bool tp3 = false;
    int64_t head0 = 0;
