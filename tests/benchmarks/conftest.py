"""Benchmark test configuration."""


def pytest_benchmark_update_machine_info(config, machine_info):
    """Normalize volatile machine metadata that changes across runs on the same host."""
    del config  # unused hook parameter

    cpu = machine_info.get("cpu")
    if not isinstance(cpu, dict):
        return

    cpu.pop("hz_actual", None)
    cpu.pop("hz_actual_friendly", None)
