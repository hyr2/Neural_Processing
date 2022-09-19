function ROI_output = readROIs(d)
    load(d);
    clear A AREA CT CT_BASELINE names rICT SC t
    ROI_output = ROI;
end