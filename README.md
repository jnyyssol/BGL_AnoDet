# BGL_AnoDet

Replication package for the paper: "Event-level Anomaly Detection on Software logs: Role of Algorithm, Threshold, and Window Size"

## Running the scripts

- There are several imports that need to be satisfied. 
- You need relatively powerful hardware to run the scripts. While the N-Gram model works well with home PC as well, the grountruth analysis would take a while.
- Download the data and add the files in the data-folder: https://zenodo.org/record/8238684
- To run the scripts adjust the path in main.py and run the configurations you wish. There are several presets, so either run the lines individually or comment some of them out.
- The results are stored in the data/results folder according to the "source" variable defined in the sheet_form_print function. There is an example file.
