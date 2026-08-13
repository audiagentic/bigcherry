import pytest

from bigcherry.device_state_validate import (
    DeviceStateEvidenceError,
    validate_device_state_report,
)


def snapshot(**overrides):
    value = {
        "status": "captured", "identity_valid": True, "hip_device": 0,
        "pci_bdf": "0000:41:00", "sclk_mhz": 1200, "mclk_mhz": 1000,
        "edge_temp_mc": 50000, "junction_temp_mc": 70000,
        "socket_power_uw": 200000000, "busy_percent": 0,
    }
    value.update(overrides)
    return value


def report(**overrides):
    value = {
        "device_state_pre": snapshot(), "device_state_post": snapshot(),
        "device_clock_drift": {
            "status": "captured", "sclk_delta_mhz": 0,
            "mclk_delta_mhz": 0, "max_abs_pct": 0.0,
        },
    }
    value.update(overrides)
    return value


def test_valid_captured_report():
    validate_device_state_report(report())


def test_unavailable_snapshot_requires_explicit_reason():
    with pytest.raises(DeviceStateEvidenceError, match="needs a reason"):
        validate_device_state_report(report(device_state_pre={"status": "unavailable"}))


def test_empty_snapshot_is_not_evidence():
    with pytest.raises(DeviceStateEvidenceError, match="missing fields"):
        validate_device_state_report(report(device_state_pre={}))


def test_zero_clock_requires_metric_unavailable_reason():
    with pytest.raises(DeviceStateEvidenceError, match="sclk_mhz=0"):
        validate_device_state_report(report(device_state_pre=snapshot(sclk_mhz=0)))


def test_zero_sensor_can_be_explicitly_unavailable():
    value = snapshot(
        edge_temp_mc=0,
        metric_status={"edge_temp_mc": "not_supported"},
        metric_reasons={"edge_temp_mc": "sensor not exposed by ASIC"},
    )
    validate_device_state_report(report(device_state_pre=value))


def test_captured_drift_requires_all_metrics():
    drift = {"status": "captured", "sclk_delta_mhz": 0}
    with pytest.raises(DeviceStateEvidenceError, match="mclk_delta_mhz"):
        validate_device_state_report(report(device_clock_drift=drift))
