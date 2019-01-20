In order to extract data from the MillionSongsDataset:

Start an EC2 Instance with the MSD EBS Volume attached to it. The links below should be helpful.
- https://gist.github.com/bwhitman/130c6290514fe4d877ff
- https://aws.amazon.com/datasets/million-song-dataset/
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-attaching-volume.html
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-public-data-sets.html#using-public-data-sets-launching-mounting

Then execute 
```
ssh -i "your_private_key_file.pem" ubuntu@ec2-1-23-456-78.compute-1.amazonaws.com
git clone https://github.com/ndey96/rec-sys.git
cd rec-sys/msd_processing
./run.sh

# Then start a new shell session on your machine
scp ubuntu@ec2-1-23-456-78.compute-1.amazonaws.com:rec-sys/msd_processing/full_msd.csv /some/local/directory
```