sudo mkdir /mnt/msd
sudo mount -t ext4 /dev/$1 /mnt/msd
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
python3 msd_to_csv.py /mnt/msd/data full_msd.csv
