-- All patients, dates and histories are fictional. CDC NDC/CVX identifiers are real.
INSERT INTO patients VALUES
('P001', 'Jordan Lee', '1962-02-10'),
('P002', 'Casey Rivera', '1958-11-20'),
('P003', 'Sam Patel', '1960-07-02'),
('P004', 'Avery Chen', '1957-04-18'),
('P005', 'Morgan Blake', '1961-08-08'),
('P006', 'Riley Park', '1964-06-13'),
('P007', 'Taylor Singh', '1956-01-06');

INSERT INTO vaccine_administrations
    (administration_id, patient_id, administered_on, ndc11, recorded_product, administration_status)
VALUES
(1,  'P001', '2024-01-10', '58160-0823-11', 'Shingrix, carton NDC', 'given'),
(2,  'P001', '2024-04-10', '58160-0828-03', 'Shingrix, vial NDC', 'given'),
(3,  'P002', '2024-05-12', '58160-0823-11', 'Shingrix', 'given'),
(4,  'P003', '2024-02-01', '58160-0823-11', 'Shingrix', 'given'),
(5,  'P003', '2024-02-15', '58160-0828-03', 'Shingrix', 'given'),
(6,  'P004', '2024-01-15', '58160-0828-03', 'Shingrix', 'given'),
(7,  'P004', '2024-04-15', '99999-9999-99', 'Unrecognized product (deliberate error)', 'given'),
(8,  'P005', '2024-01-10', '00006-4119-03', 'Gardasil 9', 'given'),
(9,  'P005', '2024-05-10', '00006-4119-01', 'Gardasil 9', 'given'),
(10, 'P006', '2024-01-01', '58160-0823-11', 'Shingrix', 'given'),
(11, 'P006', '2024-04-01', '58160-0849-43', 'Shingrix', 'entered-in-error'),
(12, 'P007', '2024-03-01', '58160-0849-43', 'Shingrix', 'given'),
(13, 'P007', '2024-06-01', '58160-0849-52', 'Shingrix', 'given');
