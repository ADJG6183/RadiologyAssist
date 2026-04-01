USE RadiologyAI;
GO

-- Stored Proc 1: sp_get_study_context
-- PURPOSE: Return study details + patient info + prior reports (for retrieval in agent pipeline)
-- INPUT: study_id
-- OUTPUT: Study + patient + prior 3 reports
DROP PROCEDURE IF EXISTS sp_get_study_context;
GO

CREATE PROCEDURE sp_get_study_context
    @study_id INT
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Return study + patient info
    SELECT
        s.study_id,
        s.study_date,
        s.modality,
        s.institution,
        p.patient_id,
        p.mrn,
        p.first_name,
        p.last_name,
        p.date_of_birth
    FROM studies s
    INNER JOIN patients p ON s.patient_id = p.patient_id
    WHERE s.study_id = @study_id;
    
    -- Return prior reports (last 3)
    SELECT TOP 3
        rd.draft_id,
        rd.study_id,
        rd.draft_text,
        rd.structured_json,
        rd.created_at,
        rd.model_name
    FROM report_drafts rd
    WHERE rd.study_id IN (
        SELECT study_id FROM studies WHERE patient_id = (
            SELECT patient_id FROM studies WHERE study_id = @study_id
        )
    )
    ORDER BY rd.created_at DESC;
END;
GO

-- Stored Proc 2: sp_insert_transcript
-- PURPOSE: Insert dictation transcript + audio URI for a study
-- INPUT: study_id, transcript_text, audio_uri
-- OUTPUT: input_id of new record
DROP PROCEDURE IF EXISTS sp_insert_transcript;
GO

CREATE PROCEDURE sp_insert_transcript
    @study_id INT,
    @transcript_text NVARCHAR(MAX),
    @audio_uri NVARCHAR(500) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    INSERT INTO report_inputs (study_id, transcript_text, audio_uri)
    VALUES (@study_id, @transcript_text, @audio_uri);
    
    SELECT @@IDENTITY AS input_id;
END;
GO

-- Stored Proc 3: sp_save_report_draft
-- PURPOSE: Save generated report draft with structured JSON
-- INPUT: study_id, draft_text, structured_json, model_name, version
-- OUTPUT: draft_id of new record
DROP PROCEDURE IF EXISTS sp_save_report_draft;
GO

CREATE PROCEDURE sp_save_report_draft
    @study_id INT,
    @draft_text NVARCHAR(MAX),
    @structured_json NVARCHAR(MAX),
    @model_name NVARCHAR(100),
    @version NVARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;
    
    INSERT INTO report_drafts (study_id, draft_text, structured_json, model_name, version)
    VALUES (@study_id, @draft_text, @structured_json, @model_name, @version);
    
    SELECT @@IDENTITY AS draft_id;
END;
GO

-- Stored Proc 4: sp_log_agent_event
-- PURPOSE: Log every step of the agent pipeline (audit trail)
-- INPUT: study_id, step, tool_name, output_summary, latency_ms
-- OUTPUT: event_id of new record
DROP PROCEDURE IF EXISTS sp_log_agent_event;
GO

CREATE PROCEDURE sp_log_agent_event
    @study_id INT,
    @step NVARCHAR(50),
    @tool_name NVARCHAR(100),
    @output_summary NVARCHAR(MAX),
    @latency_ms INT
AS
BEGIN
    SET NOCOUNT ON;
    
    INSERT INTO agent_events (study_id, step, tool_name, output_summary, latency_ms)
    VALUES (@study_id, @step, @tool_name, @output_summary, @latency_ms);
    
    SELECT @@IDENTITY AS event_id;
END;
GO

PRINT 'Stored procedures created successfully!';
