-- Create database if it doesn't exist
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'RadiologyAI')
BEGIN
    CREATE DATABASE RadiologyAI;
END
GO

USE RadiologyAI;
GO

-- Table: patients
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'patients')
BEGIN
    CREATE TABLE patients (
        patient_id INT PRIMARY KEY IDENTITY(1,1),
        mrn NVARCHAR(50) NOT NULL UNIQUE,
        first_name NVARCHAR(100) NOT NULL,
        last_name NVARCHAR(100) NOT NULL,
        date_of_birth DATE NOT NULL,
        created_at DATETIME2 DEFAULT GETUTCDATE()
    );
END
GO

-- Table: studies
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'studies')
BEGIN
    CREATE TABLE studies (
        study_id INT PRIMARY KEY IDENTITY(1,1),
        patient_id INT NOT NULL FOREIGN KEY REFERENCES patients(patient_id),
        study_date DATE NOT NULL,
        modality NVARCHAR(50) NOT NULL,
        institution NVARCHAR(200),
        created_at DATETIME2 DEFAULT GETUTCDATE()
    );
END
GO

-- Table: report_inputs
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'report_inputs')
BEGIN
    CREATE TABLE report_inputs (
        input_id INT PRIMARY KEY IDENTITY(1,1),
        study_id INT NOT NULL FOREIGN KEY REFERENCES studies(study_id),
        transcript_text NVARCHAR(MAX),
        audio_uri NVARCHAR(500),
        created_at DATETIME2 DEFAULT GETUTCDATE()
    );
END
GO

-- Table: report_drafts
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'report_drafts')
BEGIN
    CREATE TABLE report_drafts (
        draft_id INT PRIMARY KEY IDENTITY(1,1),
        study_id INT NOT NULL FOREIGN KEY REFERENCES studies(study_id),
        draft_text NVARCHAR(MAX) NOT NULL,
        structured_json NVARCHAR(MAX),
        model_name NVARCHAR(100),
        version NVARCHAR(50),
        created_at DATETIME2 DEFAULT GETUTCDATE()
    );
END
GO

-- Table: agent_events
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'agent_events')
BEGIN
    CREATE TABLE agent_events (
        event_id INT PRIMARY KEY IDENTITY(1,1),
        study_id INT NOT NULL FOREIGN KEY REFERENCES studies(study_id),
        step NVARCHAR(50),
        tool_name NVARCHAR(100),
        output_summary NVARCHAR(MAX),
        latency_ms INT,
        created_at DATETIME2 DEFAULT GETUTCDATE()
    );
END
GO

-- Create indexes for common queries
CREATE INDEX idx_studies_patient_id ON studies(patient_id);
CREATE INDEX idx_report_inputs_study_id ON report_inputs(study_id);
CREATE INDEX idx_report_drafts_study_id ON report_drafts(study_id);
CREATE INDEX idx_agent_events_study_id ON agent_events(study_id);
GO

PRINT 'Schema created successfully!';
