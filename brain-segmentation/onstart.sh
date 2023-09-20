sudo apt-get install vim screen htop -y
mkdir -p ~/dev/data/msd
cd ~/dev/data/msd/
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar .
tar -xvf Task01_BrainTumour.tar
cd ~/dev/
git clone https://github.com/JJBT/transformer_segmentation_model.git
cd transformer_segmentation_model
pip3 install -r requirements.txt
