# 🌙 Sleep Health Analytics Dashboard
### DAX Formula Documentation

A Power BI dashboard analyzing sleep patterns, stress, and physiological health indicators across a population — built on a star-schema data model with fact and dimension tables. This document catalogs every business-logic DAX formula used to power the report's KPIs and visuals.

---

## 🧠 Data Model

The report follows a standard **star schema**: one central fact table surrounded by descriptive dimension tables, joined on single-direction one-to-many relationships.

| Table | Type | Purpose |
|---|---|---|
| `Fact_SleepHealth` | Fact | Core table — one row per individual, holding sleep, stress, and vitals data |
| `Dim_AgeGroup` | Dimension | Age bracket lookup |
| `Dim_BMI` | Dimension | BMI category lookup |
| `Dim_BPCategory` | Dimension | Blood pressure category lookup |
| `Dim_Gender` | Dimension | Gender lookup |
| `Dim_HealthRiskLevel` | Dimension | Composite health risk scoring lookup |
| `Dim_Occupation` | Dimension | Occupation lookup |
| `Dim_SleepDisorder` | Dimension | Sleep disorder type lookup |
| `Dim_SleepQuality` | Dimension | Sleep quality category lookup |

---

## 📊 Core KPI Measures

Base metrics used to drive summary cards and headline statistics across the dashboard.

### `Total Individuals`
```dax
Total Individuals = COUNTA(Fact_SleepHealth[Person ID])
```
Counts the total population size in the current filter context — the base denominator for the dashboard.

### `Avg Sleep Duration`
```dax
Avg Sleep Duration = AVERAGE(Fact_SleepHealth[Sleep Duration])
```
Average nightly sleep duration (in hours) across the filtered population.

### `Avg Sleep Quality`
```dax
Avg Sleep Quality = AVERAGE(Fact_SleepHealth[Quality of Sleep])
```
Average self-reported sleep quality rating (1–10 scale).

### `Avg Sleep Debt`
```dax
Avg Sleep Debt = AVERAGE(Fact_SleepHealth[Sleep_Debt])
```
Average shortfall between recommended and actual sleep duration — a key derived indicator of sleep deprivation.

### `Avg Stress Level`
```dax
Avg Stress Level = AVERAGE(Fact_SleepHealth[Stress Level])
```
Average self-reported stress level across the filtered population.

### `Avg Health Score`
```dax
Avg Health Score = AVERAGE(Fact_SleepHealth[Health Score])
```
Average composite health score, combining multiple physiological and lifestyle indicators into a single index.

### `Sleep Disorder %`
```dax
Sleep Disorder % =
DIVIDE(
    CALCULATE(
        COUNTA(Fact_SleepHealth[Person ID]),
        Fact_SleepHealth[Has Sleep Disorder] = 1
    ),
    COUNTA(Fact_SleepHealth[Person ID]),
    0
) * 100
```
Percentage of individuals diagnosed with a sleep disorder. Uses `DIVIDE` with a zero-default to safely handle empty filter contexts and avoid division errors.

---

## 👥 Age-Group Segmentation Measures

Housed on `Dim_AgeGroup` to organize age-sliced analysis in a dedicated display folder — these break down vitals and lifestyle metrics by age bracket.

### `Average Stress`
```dax
Average Stress = AVERAGE(Fact_SleepHealth[Stress Level])
```
Average stress level within the selected age group.

### `Average Heart Rate`
```dax
Average Heart Rate = AVERAGE(Fact_SleepHealth[Heart Rate])
```
Average resting heart rate (bpm) within the selected age group.

### `Average Daily Steps`
```dax
Average Daily Steps = AVERAGE(Fact_SleepHealth[Daily Steps])
```
Average daily step count within the selected age group.

---

## 🎨 Formatted Display Measures

Presentation-layer measures that wrap base calculations in `FORMAT()` to produce clean, unit-labeled strings for cards and tooltips — keeping formatting logic centralized in DAX rather than the visual layer.

### `Sleep Duration`
```dax
Sleep Duration = FORMAT(AVERAGE('Fact_SleepHealth'[Sleep Duration]), "0.0") & " hrs"
```
Formats average sleep duration for display, e.g. `7.2 hrs`.

### `Heart Rate`
```dax
Heart Rate = FORMAT(AVERAGE('Fact_SleepHealth'[Heart Rate]), "0") & " bpm"
```
Formats average heart rate for display, e.g. `72 bpm`.

### `Sleep Quality`
```dax
Sleep Quality = FORMAT(AVERAGE('Fact_SleepHealth'[Quality of Sleep]), "0.0") & "/10"
```
Formats average sleep quality for display, e.g. `7.5/10`.

### `Stress Level`
```dax
Stress Level = FORMAT(AVERAGE('Fact_SleepHealth'[Stress Level]), "0.0") & "/10"
```
Formats average stress level for display, e.g. `5.3/10`.

### `Health Score`
```dax
Health Score = FORMAT([Avg Health Score], "0.0") & "/10"
```
Formats the composite health score for display, e.g. `8.1/10`. Built on top of the `Avg Health Score` base measure rather than recalculating the average — good practice for maintainability.

---

## 🛠 Design Notes & Best Practices Observed

- **Reusability:** Display-formatted measures (`Health Score`, `Sleep Quality`, etc.) reference underlying base measures (e.g. `[Avg Health Score]`) instead of recalculating logic — a clean, DRY approach to DAX.
- **Safe division:** `Sleep Disorder %` uses `DIVIDE()` with an explicit zero fallback instead of a raw `/` operator, preventing divide-by-zero errors in edge-case filter contexts.
- **Separation of concerns:** Raw numeric measures and their formatted string counterparts are kept as distinct measures, preserving the ability to use raw values in further calculations while offering display-ready strings for the report canvas.

---

## 📈 Summary

| Category | Count |
|---|---|
| Core KPI Measures | 7 |
| Age-Group Segmentation Measures | 3 |
| Formatted Display Measures | 5 |
| **Total Business-Logic DAX Objects** | **15** |

---

