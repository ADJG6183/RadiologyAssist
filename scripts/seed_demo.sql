USE RadiologyAI;
GO

-- Insert 2 patients
INSERT INTO patients (mrn, first_name, last_name, date_of_birth)
VALUES
    ('MRN-001', 'John', 'Doe', '1965-03-15'),
    ('MRN-002', 'Jane', 'Smith', '1978-07-22');
GO

-- Insert 6 studies (3 for John, 3 for Jane)
INSERT INTO studies (patient_id, study_date, modality, institution)
VALUES
    -- John Doe studies
    (1, '2024-01-10', 'Chest X-Ray', 'General Hospital'),
    (1, '2023-11-20', 'CT Chest', 'General Hospital'),
    (1, '2023-09-05', 'MRI Brain', 'Radiology Clinic'),
    -- Jane Smith studies
    (2, '2024-02-01', 'Abdominal Ultrasound', 'Medical Center'),
    (2, '2023-12-15', 'Mammography', 'Breast Imaging Center'),
    (2, '2023-10-10', 'Chest X-Ray', 'General Hospital');
GO

-- Insert 4 prior reports (for retrieval mock data)
INSERT INTO report_drafts (study_id, draft_text, structured_json, model_name, version)
VALUES
    -- Prior report for John Doe (study 2)
    (2, 'Small nodule in left lower lobe, recommend follow-up CT in 3 months.', 
     '{"findings": "Small nodule left lower lobe", "impression": "Possible lung lesion", "recommendation": "Follow-up CT"}',
     'MockLLMClient', '1.0.0'),
    -- Prior report for John Doe (study 3)
    (3, 'No acute intracranial abnormality. Normal brain parenchyma.',
     '{"findings": "Normal brain MRI", "impression": "No acute findings", "recommendation": "None"}',
     'MockLLMClient', '1.0.0'),
    -- Prior report for Jane Smith (study 5)
    (5, 'BI-RADS 2. Normal breast tissue, no suspicious masses.',
     '{"findings": "Normal breast tissue", "impression": "Negative mammography", "recommendation": "Routine screening"}',
     'MockLLMClient', '1.0.0'),
    -- Prior report for Jane Smith (study 6)
    (6, 'Clear lung fields, normal cardiac silhouette, no findings.',
     '{"findings": "Clear lungs", "impression": "Normal chest X-Ray", "recommendation": "None"}',
     'MockLLMClient', '1.0.0');
GO

PRINT 'Seed data inserted successfully!';
PRINT 'Patients: 2';
PRINT 'Studies: 6';
PRINT 'Prior Reports: 4';
