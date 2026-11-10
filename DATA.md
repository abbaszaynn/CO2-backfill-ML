# Data

## Source

The data were provided by Zahid Hussain's research group and belong to the CO2-injected gold-tailings
cement paste backfill (CGCPB) programme described in:

> Hussain, Z., Fu, J., Li, J., Li, Y. (2025). CO2 injection stirring through slurry to enhance performance of gold
> tailings based sustainable cement paste backfill. *Journal of Cleaner Production* 518, 145950.
> https://doi.org/10.1016/j.jclepro.2025.145950

The materials, mixing, CO2 injection set-up and porosity test (low-field NMR) are described in that paper.

**Important:** this is a *preliminary* file. Its records are not independent specimen measurements: they follow
a reference curve with fixed factors for cement ratio, stirring speed and flow rate, contain no replicates, and
include values filled in between and beyond the tested injection times (see the paper, results section "Preliminary dataset and pre-processing").
The analysis will be repeated on the specimen-level laboratory data when it is available.

## Files

| File | What it is |
|---|---|
| `data/raw_preliminary_dataset.xlsx` | The file as received (sheet `Cleaned_Data`: 500 records, 21 columns; sheets `Cleaning_Summary` and `Feature_Definitions` describe how it was prepared before we received it). |
| `co2_backfill_train.csv` | The data used for every model in the paper and the app: 786 records, 81 recipes. Rebuilt exactly by `python paper/clean_data.py`. |

## Cleaning steps (`paper/clean_data.py`)

1. **Repaired the cement-to-tailings ratio.** The spreadsheet stored 1:4 as the time `01:04:00`; the numeric column `Cement_Ratio` (4, 6, 8) is used instead.
2. **Removed 14 engineered columns** (log, square root, square, inverse, products, z-score and the outlier flag). They duplicate the inputs, and the inverse of injection time was set to 10^9 at 0 min. No outliers had been flagged (|z| > 3: none).
3. **Kept only the 10 injection times tested in the source programme** (0, 0.5, 1, 1.5, 2, 5, 8, 10, 15, 20 min). Removed 2.5, 3, 12, 14 and 18 min (interpolated between tested times) and 22, 25, 28 and 30 min (outside the tested range).
4. **Combined porosity at 3, 7 and 14 days into one response** with curing age as an input, so that no response is used to predict another.

Result: 786 records (262 conditions x 3 curing ages), 81 recipes, no missing values or duplicates.

## Columns of `co2_backfill_train.csv`

| Column | Unit | Meaning | Values |
|---|---|---|---|
| `recipe` | - | cement ratio, stirring speed, flow rate and curing age, e.g. `1:6_800rpm_1Lmin_14d` | 81 recipes |
| `cement_ratio` | - | cement-to-tailings ratio | 1:4, 1:6, 1:8 |
| `cement_fraction` | - | cement mass fraction of solids = 1 / (1 + ratio) | 0.2, 0.143, 0.111 |
| `stirring_rpm` | rpm | stirring speed during injection | 500, 700, 800 |
| `co2_flow_L_min` | L/min | CO2 flow rate | 1, 2, 3 |
| `curing_days` | days | curing age | 3, 7, 14 |
| `co2_time_min` | min | CO2 injection time | 0, 0.5, 1, 1.5, 2, 5, 8, 10, 15, 20 |
| `porosity_pct` | % | porosity (LF-NMR) | 7.80 to 13.63 |

## Licence

*(To be agreed with Z. Hussain before the repository is made public, e.g. CC BY 4.0 with citation of Hussain et al., 2025.)*
