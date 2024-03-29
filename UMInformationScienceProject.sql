USE UDT4Archive;
SET NOCOUNT ON;
/*
David Kulpanowski
dkulpanowski@leegov.com
239-910-7968
20 February 2019
U of M students from Information Sciences are building a model for fire susceptibility.
Exclude medical calls.
*/
SELECT 
  I_Agency AS [FireAgency1]
, I_EventNumber AS [UniqueEventNumber]
, RTRIM(I_Address) AS [IncidentAddress]
, I_MapX AS [Longitude]
, I_MapY AS [Latitude]
, RTRIM(I_LocationText) AS [IncidentLocationDescription]
, RTRIM(I_CrossStreet1) AS [CrossStreet1]
, RTRIM(I_CrossStreet2) AS [CrossStreet2]
, RTRIM(I_ApartmentNumber) AS [ApartmentNumber]
, I_tTimeCreate AS [TimeIncidentCreated]
, I_tTimeDispatch AS [TimeCrewNotifiedOfIncident]
, I_tTimeArrival AS [TimeCrewArrivesAtIncident]
, I_tTimeClosed AS [TimeIncidentClosed]
, RTRIM(ILO_City) AS [City]
, RTRIM(ILO_AreaAgency) AS [FireAgency]
, RTRIM(ITI_TypeID) AS [IncidentTypeCode]
, RTRIM(ITI_TypeText) AS [IncidentTypeDescription]
, RTRIM(IDI_DispositionCode_1) AS [DispositionCode]
, RTRIM(IDI_DispositionText_1) AS [DispositionDescription]
, RTRIM(PUN_UnitAgency) AS [FireAgency]
, RTRIM(PUN_UnitID) AS [EmergencyApparatus]
FROM UDT4Archive.dbo.IIncident
--WHERE I_Agency = 'FM' /*cannot use I_Agency because the event numbers were changed and agency became CW */
WHERE PUN_UnitAgency = 'FM'
AND (
ITI_TypeText LIKE 'Structure Fire%' OR ITI_TypeText LIKE 'Fire Alarm%' OR ITI_TypeText LIKE 'Outside Fire%' 
OR ITI_TypeText LIKE 'Electrical hazard%' OR ITI_TypeText LIKE 'Smoke%' OR ITI_TypeText LIKE 'Gas Leak%' 
OR ITI_TypeText LIKE 'Fuel Spill%' OR ITI_TypeText LIKE 'Electrical Hazard%' OR ITI_TypeText LIKE 'Alarms%'
)
AND I_tTimeClosed > 'january 1, 2008'
ORDER BY I_EventNumber ASC



--SELECT
--  RTRIM(ITI_TypeText) AS [IncidentTypeDescription]
--, COUNT(I_EventNumber) AS [VolumeOfThisType]
--FROM IIncident
--WHERE PUN_UnitAgency = 'FM'
--GROUP BY ITI_TypeText
--ORDER BY ITI_TypeText
--;


--SELECT DATEDIFF(YEAR, 'january 1, 2008', 'february 20, 2019')