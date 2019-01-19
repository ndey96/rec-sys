In order to extract data from the MillionSongsDataset:

```
Copy MSD to this directory
python msd_to_csv.py /mnt/snap/data full_msd.csv
python count_lines.py
scp song_data.csv
```