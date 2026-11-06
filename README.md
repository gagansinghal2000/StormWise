# StormWise — River Aire / Yorkshire Water prototype

## IMPORTANT: use this build for the real map
This build replaces the previous offline coordinate plot with real Yorkshire Water 2025 EDM overflow points and an OpenStreetMap basemap.

## Run
1. Extract this ZIP into a NEW folder (do not copy only app.py into an old folder).
2. Open a terminal in the extracted `StormWise_Prototype` folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.

## Map
- The map fetches Yorkshire Water 2025 EDM point records from the Rivers Trust public ArcGIS FeatureServer.
- It filters points to a geographic window around the River Aire corridor.
- Basemap style is `open-street-map`.
- Internet access is required on the computer running Streamlit.
- If the map still shows longitude/latitude axes and a pale green plot, the old offline `app.py` is still being run.

## Real vs synthetic
Real: mapped overflow points, 2025 counted spills and total duration from the EDM feature layer.
Synthetic: intervention options, costs, benefits, buildability and community scores, and portfolio outputs.
The study window is a bounding rectangle, not an official River Aire catchment boundary. The map does not yet join the actual EDM assets to the synthetic intervention options.
