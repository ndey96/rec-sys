# How to extract data from the full MillionSongsDataset

Start an EC2 Instance (I used an m5.2xlarge instance) with the Ubuntu 18.04 AMI in us-east-1a with the MSD EBS Volume attached to it. The links below should be helpful.
- https://gist.github.com/bwhitman/130c6290514fe4d877ff
- https://aws.amazon.com/datasets/million-song-dataset/
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-attaching-volume.html
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-public-data-sets.html#using-public-data-sets-launching-mounting

Execute 'ssh -i "your_private_key_file.pem" ubuntu@ec2-123.compute-1.amazonaws.com'
Run 'lsblk' to find the device name of the MSD volume you attached to the instance.
```
ubuntu@ip-172-31-25-116:~/rec-sys/msd_processing$ lsblk
NAME        MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
loop0         7:0    0 87.9M  1 loop /snap/core/5328
loop1         7:1    0 12.7M  1 loop /snap/amazon-ssm-agent/495
nvme0n1     259:0    0    8G  0 disk 
└─nvme0n1p1 259:1    0    8G  0 part /
nvme1n1     259:2    0  500G  0 disk /mnt/msd
```
From the above output we know that the MSD volume name is nvme1n1.

```
git clone https://github.com/ndey96/rec-sys.git
cd rec-sys/msd_processing
tmux new
./run.sh nvme1n1
Type "ctrl+b d" to detach from the tmux session
tmux attach-session -t 0
```

Then start a new shell session on your local machine and run `scp -i your_private_key_file.pem ubuntu@ec2-123.compute-1.amazonaws.com:rec-sys/msd_processing/full_msd.csv /some/local/directory` to grab the processed csv file.