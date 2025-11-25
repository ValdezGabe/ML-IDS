"""
Configuration management for ML-IDS
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for ML-IDS project"""

    # Project paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    FEATURES_DIR = DATA_DIR / "features"
    TABLEAU_EXPORT_DIR = DATA_DIR / "tableau_exports"

    MODEL_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"

    # Dataset configuration
    UNSW_NB15_PATH = os.getenv("UNSW_NB15_PATH", str(RAW_DATA_DIR / "UNSW-NB15"))

    # UNSW-NB15 feature columns
    UNSW_FEATURES = [
        'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
        'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss',
        'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
        'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
        'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
        'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
        'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
        'ct_srv_dst', 'is_sm_ips_ports'
    ]

    # Attack categories for UNSW-NB15
    ATTACK_CATEGORIES = {
        'Normal': 0,
        'Generic': 1,
        'Exploits': 2,
        'Fuzzers': 3,
        'DoS': 4,
        'Reconnaissance': 5,
        'Analysis': 6,
        'Backdoor': 7,
        'Shellcode': 8,
        'Worms': 9
    }

    NUM_CLASSES = len(ATTACK_CATEGORIES)

    # LSTM Model configuration
    LSTM_CONFIG = {
        'sequence_length': 10,  # Number of time steps
        'lstm_units': [128, 64],  # LSTM layer units
        'dense_units': [64, 32],  # Dense layer units
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 50,
        'validation_split': 0.2
    }

    # Tableau configuration
    TABLEAU_SERVER_URL = os.getenv("TABLEAU_SERVER_URL", "")
    TABLEAU_SITE_ID = os.getenv("TABLEAU_SITE_ID", "")
    TABLEAU_USERNAME = os.getenv("TABLEAU_USERNAME", "")
    TABLEAU_PASSWORD = os.getenv("TABLEAU_PASSWORD", "")
    TABLEAU_TOKEN_NAME = os.getenv("TABLEAU_TOKEN_NAME", "")
    TABLEAU_TOKEN_VALUE = os.getenv("TABLEAU_TOKEN_VALUE", "")

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
                        cls.FEATURES_DIR, cls.TABLEAU_EXPORT_DIR,
                        cls.MODEL_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_attack_name(cls, label):
        """Get attack name from label"""
        for name, idx in cls.ATTACK_CATEGORIES.items():
            if idx == label:
                return name
        return "Unknown"
