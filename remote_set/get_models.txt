mkdir -p models/neuralgcm models/aifs models/aurora

# NeuralGCM models
curl -fL "https://storage.cloud.google.com/neuralgcm/models/v1/stochastic_1_4_deg.pkl" \
  -o "models/neuralgcm/stochastic_1_4_deg.pkl"

curl -fL "https://storage.cloud.google.com/neuralgcm/models/v1_precip/stochastic_precip_2_8_deg.pkl" \
  -o "models/neuralgcm/stochastic_precip_2_8_deg.pkl"

curl -fL "https://storage.cloud.google.com/neuralgcm/models/v1_precip/stochastic_evap_2_8_deg.pkl" \
  -o "models/neuralgcm/stochastic_evap_2_8_deg.pkl"

curl -fL "https://storage.cloud.google.com/neuralgcm/models/v1/deterministic_0_7_deg.pkl" \
  -o "models/neuralgcm/deterministic_0_7_deg.pkl"

curl -fL "https://storage.cloud.google.com/neuralgcm/models/v1/deterministic_1_4_deg.pkl" \
  -o "models/neuralgcm/deterministic_1_4_deg.pkl"

curl -fL "https://storage.cloud.google.com/neuralgcm/models/v1/deterministic_2_8_deg.pkl" \
  -o "models/neuralgcm/deterministic_2_8_deg.pkl"


# AIFS models
curl -fL "https://huggingface.co/ecmwf/aifs-single-1.1/blob/main/aifs-single-mse-1.1.ckpt" \
  -o "models/aifs/aifs-single-mse-1.1.ckpt"

curl -fL "https://huggingface.co/ecmwf/aifs-ens-1.0/blob/main/aifs-ens-crps-1.0.ckpt" \
  -o "models/aifs/aifs-ens-crps-1.0.ckpt"


# MS Aurora models
curl -fL "https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-finetuned.ckpt" \
  -o "models/aurora/aurora-0.25-finetuned.ckpt"