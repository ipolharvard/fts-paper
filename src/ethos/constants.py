from enum import StrEnum
from importlib.resources import files
from pathlib import Path

MAPPINGS_DIR: Path = files("ethos.tokenize.maps")
PROJECT_ROOT: Path = (MAPPINGS_DIR / "../../../..").resolve()

PATIENT_ID_COL = "subject_id"


class SpecialToken(StrEnum):
    DOB = "MEDS_BIRTH"
    DEATH = "MEDS_DEATH"
    TIMELINE_START = "TIMELINE_START"
    TIMELINE_END = "TIMELINE_END"

    # hospital adm and dc
    ADMISSION = "HOSPITAL_ADMISSION"
    DISCHARGE = "HOSPITAL_DISCHARGE"

    # only MIMIC
    ICU_ADMISSION = "ICU_ADMISSION"
    ICU_DISCHARGE = "ICU_DISCHARGE"
    ED_ADMISSION = "ED_REGISTRATION"
    ED_DISCHARGE = "ED_OUT"
    SOFA = "SOFA"


STATIC_DATA_FN = "static_data.parquet"
