echo "[*] Running master script on tpu-test-0..."; \
gcloud compute tpus tpu-vm ssh tpu-test-0 \
       --worker=all \
       --zone=us-central2-b \
       --command='export PATH=~/miniconda/bin:$PATH; \
       source $HOME/miniconda/bin/activate; \
       conda activate autocurricula; \
       python3 tpu_chess_league;';