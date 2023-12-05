echo "[*] Copying codebase to tpu-test-0";
gcloud alpha compute tpus tpu-vm scp ./tpu_chess_league.py tpu-test-0:~/ \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b;
gcloud alpha compute tpus tpu-vm scp ../autocurricula/ tpu-test-0:~/autocurricula \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b \
       --recurse;
# Repeated so as to avoid first-time ssh handshake issues.
gcloud alpha compute tpus tpu-vm scp ./tpu_chess_league.py tpu-test-0:~/ \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b;
gcloud alpha compute tpus tpu-vm scp ../setup.py tpu-test-0:~/ \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b;
gcloud alpha compute tpus tpu-vm scp ../environment.yml tpu-test-0:~/ \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b;

# TODO: Add wandb key transfer and init.
echo "[*] Procuring tpu-test-0...";
gcloud alpha compute tpus tpu-vm ssh tpu-test-0 \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b \
       --command='wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh; \
       chmod +x ~/miniconda.sh; \
       ~/miniconda.sh -b -p ~/miniconda; \
       export PATH=~/miniconda/bin:$PATH; \
       conda init bash; \
       conda update -n base -c defaults conda -y; \
       source $HOME/miniconda/bin/activate; \
       conda env create -f environment.yml; \
       conda activate autocurricula; \
       pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html ;
';