# CrossSectionEditor
This plugin for QGIS provides an interactive tool for editing and managing cross-section data. Primarily for use with ESTRY-TUFLOW cross sections.

![image](https://github.com/user-attachments/assets/35af0631-a13b-4c0f-a7bf-77ba1e60ae16)

Features:
 - Load in multiple CSV files to show the plot and table view side by side.
 - Quickly navigate through sections using Previous/Next buttons or clicking on the file name.
 - Specify the column headers in the 'Column Settings'. Can also right-click on the table view to set the column for current view only.
 - Quickly trim sections on the **left** / **right** bank by **Ctrl** / **Alt** clicking a point on the plot or right clicking a row in the table view.
 - Specify how you would like to save new files as - Increment version by appending `_<VERSION>` to the filename or changing in-place.
 - Load in a `.shp` or `.gpkg` polygon shapefile. If the CSV files have WKT point text column - overlap between CSV section and polygon will be shown.

Options:
 - `Fix Verticals and Order` checkbox - automatically ensure that all X values increase by atleast 0.001 when loading and saving sections.
 - `Open and save with StartX=0` checkbox - automatically ensure that the left most X value of section on load or on save is equal to 0.
 - `Autosave on section change` checkbox - automatically save the CSV file inplace or with new version suffix on section view change regardless if any changes have been made.
 - `Make plot file on save` checkbox - automatically save the current plot along side the file with same filename but `.png` extension.
