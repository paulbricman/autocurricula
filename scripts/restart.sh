gcloud alpha compute tpus queued-resources delete queued-resource-0 \
--project the-chronicles-of-computation \
--zone us-central2-b \
--force \
--async

echo "[*] Creating TPU Pod...";
gcloud alpha compute tpus queued-resources create queued-resource-0 \
--node-id tpu-test-0 \
--project the-chronicles-of-computation \
--zone us-central2-b \
--accelerator-type v4-8 \
--runtime-version tpu-vm-v4-pt-2.0 \
--best-effort;

gcloud alpha compute tpus queued-resources list --project the-chronicles-of-computation \
--zone us-central2-b;