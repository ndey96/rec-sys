# Data

This directory contains scripts to process data and csv files which can be used in recommender system pipelines.

### Guide
- `triplets.csv` -> contains user,song,play_count triplets from the [Echo Nest Taste Profile Subset](https://labrosa.ee.columbia.edu/millionsong/tasteprofile)
- `mismatches.csv` -> contains data about [mismatched songs in the MSD](https://labrosa.ee.columbia.edu/millionsong/blog/12-2-12-fixing-matching-errors)
- `full_msd.csv` -> contains song metadata from the full million songs dataset (~1M songs)
- `msd_subset_metadata.csv` -> contains song metadata from the million songs subset dataset (~10k songs)


FYI all these scripts are kind of inefficient and should be refactored with
```
import csv
csv_columns = ['No','Name','Country']
dict_data = [
{'No': 1, 'Name': 'Alex', 'Country': 'India'},
{'No': 2, 'Name': 'Ben', 'Country': 'USA'},
{'No': 3, 'Name': 'Shri Ram', 'Country': 'India'},
{'No': 4, 'Name': 'Smith', 'Country': 'USA'},
{'No': 5, 'Name': 'Yuva Raj', 'Country': 'India'},
]
csv_file = "Names.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
except IOError:
    print("I/O error") 
```