PRAGMA foreign_keys = ON;
CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    date_of_birth TEXT NOT NULL
);

CREATE TABLE vaccine_administrations (
    administration_id INTEGER PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    administered_on TEXT NOT NULL,
    ndc11 TEXT NOT NULL,  -- CDC 5-4-2 representation; preserve leading zeroes
    recorded_product TEXT NOT NULL,
    administration_status TEXT NOT NULL DEFAULT 'given'
        CHECK (administration_status IN ('given', 'entered-in-error'))
);
CREATE INDEX ix_admin_patient_date ON vaccine_administrations(patient_id, administered_on);
CREATE INDEX ix_admin_ndc ON vaccine_administrations(ndc11);

-- Terminology lookup is modeled separately. We never modify the source NDC.
CREATE TABLE ndc_cvx_map (
    ndc11 TEXT PRIMARY KEY,
    cvx_code TEXT,
    cvx_description TEXT,
    vaccine_group TEXT,
    mapping_status TEXT NOT NULL CHECK (mapping_status IN ('mapped','unmapped')),
    mapping_source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    CHECK ((mapping_status = 'mapped' AND cvx_code IS NOT NULL)
        OR (mapping_status = 'unmapped' AND cvx_code IS NULL))
);
