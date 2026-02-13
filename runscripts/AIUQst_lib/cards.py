"""
Author: Jacopo Grassi
Institution: Politecnico di Torino
Email: jacopo.grassi@polito.it

Created: 2025-01-12
Last modified: 2025-02-13

Description:
    
"""   

# Built-in/Generics
import logging
import yaml
import os

# Third party

# Local

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_model_card(hpcrootdir: str, model: str) -> list:

    yaml_path = os.path.abspath(f"{hpcrootdir}/conf/cards/models/{model}_card.yml")
    print(f"Reading model card from: {yaml_path}")
    with open(yaml_path, "r") as f:
        model_card = yaml.safe_load(f)

    return model_card

def read_ic_card(hpcrootdir: str, ic: str) -> list:

    yaml_path = os.path.abspath(f"{hpcrootdir}/conf/cards/ics/{ic}_card.yml")
    print(f"Reading ic card from: {yaml_path}")
    with open(yaml_path, "r") as f:
        ic_card = yaml.safe_load(f)
    
    return ic_card

def read_std_version(hpcrootdir: str, version: str):

    yaml_path = os.path.join(f'{hpcrootdir}/conf/AIUQ-st', version)
    print(f"Reading standard version from: {yaml_path}")

    standard_dict = {}

    pressure_levels_file = os.path.join(yaml_path, 'pressure_levels.yml')
    with open(pressure_levels_file, "r") as f:
        pressure_levels_std = yaml.safe_load(f)
    
    variables_file = os.path.join(yaml_path, 'variables.yml')
    with open(variables_file, "r") as f:
        variables_std = yaml.safe_load(f)
    
    standard_dict['pressure_levels'] = pressure_levels_std
    standard_dict['variables'] = variables_std
    
    return standard_dict
