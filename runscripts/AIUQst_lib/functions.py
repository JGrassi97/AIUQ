import argparse
import yaml
import ast
import os
from typing import Any, List


def normalize_out_vars(v: Any) -> List[str]:
    """
    Converte out_vars in una lista di stringhe in modo robusto.
    Accetta:
      - lista già pronta: ['temperature']
      - stringa python-list: "['temperature']"
      - stringa singola: "temperature"
      - stringa separata da spazi/virgole: "t q u,v"
    """
    if v is None:
        return []

    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []

        # caso: "['temperature', 'u']"
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        # caso: "temperature, u, v" oppure "temperature u v"
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        if " " in s:
            return [p.strip() for p in s.split() if p.strip()]

        # caso: "temperature"
        return [s]

    # fallback
    return [str(v).strip()]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run NeuralGCM model')
    parser.add_argument('--config', '-c', type=str, default='config_neuralgcm.yaml',
                        help='Path to the config file')
    return parser.parse_args()

def read_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def define_variables(config):
    model_checkpoint = config['model_checkpoint']
    INI_DATA_PATH = config['INI_DATA_PATH']
    start_time = config['start_time']
    end_time = config['end_time']
    data_inner_steps = config['data_inner_steps']
    inner_steps = config['inner_steps']

    rk = config['rng_key']
    rng_key = 1 if rk in (None, "", "None") else int(rk)

    output_path = config['output_path']
    output_vars = normalize_out_vars(config.get('out_vars'))
    ics_temp_dir = config['ics_temp_dir']
    static_data = config['static_data']

    return (model_checkpoint, INI_DATA_PATH, start_time, end_time,
            data_inner_steps, inner_steps, rng_key, output_path,
            output_vars, ics_temp_dir, static_data)