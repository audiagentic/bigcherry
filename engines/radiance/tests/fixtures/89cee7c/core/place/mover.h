class Residency final : public ResidencyTable {
public:
    void publish(rad_weight w, void* ptr);
    void begin_transfer(rad_weight w, void* ptr, RadEvent ready);
};
