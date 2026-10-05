# SmartCrop AI — Fixed Version

## Fix in this version
The earlier version expected columns named `N`, `P`, and `K` but its column-normalization logic did not correctly map the common CSV names `N`, `P`, and `K`. This version explicitly supports:
- N / nitrogen
- P / phosphorus
- K / potassium
- temperature
- humidity
- ph
- rainfall
- label / crop

It also checks downloaded ZIP contents and automatically selects a compatible CSV.

## Run
```powershell
cd "C:\Users\HP\Downloads\SmartCropAI_Paper_Implementation_FIXED"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If you already have a `Crop_recommendation.csv`, keep it beside `app.py`.

## Research paper
Dey, B., Ferdous, J., & Ahmed, R. (2024). Machine learning based recommendation of agricultural and horticultural crop farming in India under the regime of NPK, soil pH and three climatic variables. Heliyon, 10, e25112.
