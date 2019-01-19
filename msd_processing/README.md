In order to extract data from the MillionSongsDataset:

```
Copy MSD to this directory
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
python3 msd_to_csv.py /mnt/snap/data full_msd.csv
python count_lines.py
scp song_data.csv
```