from mspf_net.config_utils import (
    get_active_dataset_ids,
    get_config_list,
    get_primary_scratch_dataset_ids,
)


ACTIVE_RAW_DATASETS: list[int] = [int(v) for v in get_config_list("datasets", "active_raw", default=[1, 2, 4, 5, 8])]
ACTIVE_THESIS_DATASETS: list[int] = get_active_dataset_ids()
PRIMARY_SCRATCH_DATASETS: list[int] = get_primary_scratch_dataset_ids()

DATASET_NAMES: dict[int, str] = {
    1: "D1_BearingTV",
    2: "D2_PlanetaryGB",
    3: "D3",
    31: "D3C1_CWRU_Bearing",
    39: "D3C9_Gearbox",
    4: "D4_MultiModeGB",
    5: "D5_MixedBG",
    6: "D6_GearCrack",
    7: "D7_GearboxBench",
    8: "D8_HUSTGearbox",
}

DATASET_DISPLAY: dict[int, str] = {
    1: "D1",
    2: "D2",
    3: "D3",
    31: "D3C1",
    39: "D3C9",
    4: "D4",
    5: "D5",
    6: "D6",
    7: "D7",
    8: "D8",
}

# Native sampling rates (Hz) for each dataset ID.
# Both eda_utils (FS_MAP) and preprocess_utils (FS_NATIVE) import from here
# so the two never diverge.
FS_NATIVE: dict[int, int] = {
    1: 200_000,
    2:  48_000,
    3:  12_000,
    31: 12_000,
    39: 12_000,
    4:  10_000,
    5:  10_000,
    6:  10_000,
    7:  10_000,
    8:  25_600,
}


def get_dataset_name(ds_id: int) -> str:
    return DATASET_NAMES.get(int(ds_id), f"D{ds_id}")


def get_dataset_display(ds_id: int) -> str:
    return DATASET_DISPLAY.get(int(ds_id), f"D{ds_id}")


def get_fs_native(ds_id: int) -> int:
    """Return the native sampling rate for *ds_id*, with a clear error on miss."""
    try:
        return FS_NATIVE[ds_id]
    except KeyError:
        valid = sorted(FS_NATIVE)
        raise ValueError(
            f"Unknown dataset ID {ds_id!r}. "
            f"Valid IDs are: {valid}"
        ) from None
