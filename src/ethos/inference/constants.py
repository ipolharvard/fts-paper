from enum import StrEnum


class Task(StrEnum):
    HOSPITAL_MORTALITY = "hospital_mortality"
    READMISSION = "readmission"
    DRG_PREDICTION = "drg"
    SOFA_PREDICTION = "sofa"
    ICU_MORTALITY = "icu_mortality"
    ICU_READMISSION = "icu_readmission"
    ICU_ADMISSION = "icu_admission"
    SYNTHETIC = "synthetic"


class Reason(StrEnum):
    GOT_TOKEN = "token_of_interest"
    KEY_ERROR = "key_error"
    TIME_LIMIT = "time_limit"
