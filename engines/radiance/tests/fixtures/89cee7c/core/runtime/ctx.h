class Ctx : public RadCtx {
public:
    void* weight_ptr(rad_weight w);
    void* buf_ptr(rad_buf b);

    int route_report(int layer, const RadRouting* r);
private:
    ResidencyTable* residency_ = nullptr;
};
