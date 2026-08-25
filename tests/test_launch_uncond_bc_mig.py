from experiments.libero import launch_uncond_bc_mig as launcher


def test_rank_environment_isolates_one_mig_at_local_device_zero() -> None:
    environment = launcher._rank_environment(
        {"KEEP": "yes", "CUDA_VISIBLE_DEVICES": "old"},
        rank=2,
        world_size=4,
        mig_uuid="MIG-example",
        master_addr="127.0.0.1",
        master_port=29471,
    )

    assert environment == {
        "KEEP": "yes",
        "CUDA_VISIBLE_DEVICES": "MIG-example",
        "RANK": "2",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "4",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29471",
    }
