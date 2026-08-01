from experiments.runners.launch_colab import requested_hardware, session_hardware


def test_session_hardware_parses_colab_status() -> None:
    output = (
        "[mindscopex] gpu-host.example | Hardware: H100 | Variant: GPU | Status: IDLE\n"
    )

    assert session_hardware(output) == "H100"


def test_session_hardware_returns_none_for_missing_session() -> None:
    assert session_hardware("[colab] Session 'mindscopex' not found.\n") is None


def test_requested_hardware_prefers_gpu_then_tpu_then_cpu() -> None:
    assert requested_hardware("a100", None) == "A100"
    assert requested_hardware(None, "v6e1") == "V6E1"
    assert requested_hardware(None, None) == "CPU"
