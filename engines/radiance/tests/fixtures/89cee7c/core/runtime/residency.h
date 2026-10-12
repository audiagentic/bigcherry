struct WeightSlot {
    void*    ptr;
    RadEvent ready;
    uint32_t generation;
};

class ResidencyTable {
public:
    const WeightSlot* lookup(rad_weight w) const;
};
