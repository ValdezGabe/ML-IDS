"""
Generate synthetic UNSW-NB15-like data for testing
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config import Config


def generate_synthetic_unsw_data(n_samples=50000, random_state=42):
    """
    Generate synthetic network traffic data mimicking UNSW-NB15 format

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed

    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(random_state)

    print(f"Generating {n_samples} synthetic network traffic records...")

    # Define attack categories
    attack_categories = list(Config.ATTACK_CATEGORIES.keys())
    attack_weights = [0.6, 0.05, 0.08, 0.04, 0.08, 0.05, 0.03, 0.03, 0.03, 0.01]

    # Generate basic features
    data = {
        # Connection features
        'dur': np.random.exponential(scale=10, size=n_samples),
        'spkts': np.random.poisson(lam=50, size=n_samples),
        'dpkts': np.random.poisson(lam=40, size=n_samples),
        'sbytes': np.random.exponential(scale=1000, size=n_samples),
        'dbytes': np.random.exponential(scale=800, size=n_samples),

        # Protocol and state
        'proto': np.random.choice(['tcp', 'udp', 'icmp'], size=n_samples, p=[0.7, 0.2, 0.1]),
        'service': np.random.choice(['http', 'ftp', 'ssh', 'dns', 'smtp', '-'],
                                   size=n_samples, p=[0.3, 0.1, 0.1, 0.2, 0.1, 0.2]),
        'state': np.random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST'],
                                 size=n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.1]),

        # TTL and loss
        'sttl': np.random.randint(32, 128, size=n_samples),
        'dttl': np.random.randint(32, 128, size=n_samples),
        'sloss': np.random.poisson(lam=0.5, size=n_samples),
        'dloss': np.random.poisson(lam=0.5, size=n_samples),

        # Load and window
        'sload': np.random.exponential(scale=100, size=n_samples),
        'dload': np.random.exponential(scale=80, size=n_samples),
        'swin': np.random.randint(0, 255, size=n_samples),
        'dwin': np.random.randint(0, 255, size=n_samples),

        # TCP base sequence numbers
        'stcpb': np.random.randint(0, 2**32, size=n_samples),
        'dtcpb': np.random.randint(0, 2**32, size=n_samples),

        # Mean packet size
        'smeansz': np.random.exponential(scale=400, size=n_samples),
        'dmeansz': np.random.exponential(scale=350, size=n_samples),

        # Transaction depth
        'trans_depth': np.random.randint(0, 10, size=n_samples),
        'res_bdy_len': np.random.exponential(scale=500, size=n_samples),

        # Jitter
        'sjit': np.random.exponential(scale=5, size=n_samples),
        'djit': np.random.exponential(scale=5, size=n_samples),

        # Inter-packet time
        'sintpkt': np.random.exponential(scale=10, size=n_samples),
        'dintpkt': np.random.exponential(scale=10, size=n_samples),

        # TCP features
        'tcprtt': np.random.exponential(scale=50, size=n_samples),
        'synack': np.random.exponential(scale=30, size=n_samples),
        'ackdat': np.random.exponential(scale=20, size=n_samples),

        # Binary features
        'is_sm_ips_ports': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'is_ftp_login': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),

        # Connection counts
        'ct_state_ttl': np.random.randint(1, 20, size=n_samples),
        'ct_flw_http_mthd': np.random.randint(0, 10, size=n_samples),
        'ct_ftp_cmd': np.random.randint(0, 5, size=n_samples),
        'ct_srv_src': np.random.randint(1, 50, size=n_samples),
        'ct_srv_dst': np.random.randint(1, 50, size=n_samples),
        'ct_dst_ltm': np.random.randint(1, 30, size=n_samples),
        'ct_src_ltm': np.random.randint(1, 30, size=n_samples),
        'ct_src_dport_ltm': np.random.randint(1, 20, size=n_samples),
        'ct_dst_sport_ltm': np.random.randint(1, 20, size=n_samples),
        'ct_dst_src_ltm': np.random.randint(1, 40, size=n_samples),

        # Placeholder for IP addresses and ports
        'srcip': [f"192.168.{np.random.randint(0,255)}.{np.random.randint(1,255)}"
                 for _ in range(n_samples)],
        'sport': np.random.randint(1024, 65535, size=n_samples),
        'dstip': [f"10.0.{np.random.randint(0,255)}.{np.random.randint(1,255)}"
                 for _ in range(n_samples)],
        'dsport': np.random.randint(1, 65535, size=n_samples),

        # Time features
        'stime': np.arange(n_samples) + np.random.randint(0, 1000000),
        'ltime': np.arange(n_samples) + np.random.randint(1000000, 2000000),
    }

    df = pd.DataFrame(data)

    # Ensure numeric columns are proper numeric types
    numeric_cols = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl',
                    'sloss', 'dloss', 'sload', 'dload', 'swin', 'dwin', 'stcpb',
                    'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
                    'sjit', 'djit', 'sintpkt', 'dintpkt', 'tcprtt', 'synack',
                    'ackdat', 'is_sm_ips_ports', 'is_ftp_login', 'ct_state_ttl',
                    'ct_flw_http_mthd', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
                    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                    'ct_dst_src_ltm', 'sport', 'dsport']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Generate attack categories and labels
    attack_cats = np.random.choice(attack_categories, size=n_samples, p=attack_weights)
    df['attack_cat'] = attack_cats
    df['label'] = (df['attack_cat'] != 'Normal').astype(int)

    # Add attack-specific patterns
    for i, attack in enumerate(attack_categories[1:], 1):  # Skip 'Normal'
        attack_mask = df['attack_cat'] == attack

        if attack == 'DoS':
            # DoS: High packet rates, high bytes
            df.loc[attack_mask, 'spkts'] *= 5
            df.loc[attack_mask, 'sbytes'] *= 3

        elif attack == 'Reconnaissance':
            # Recon: Many connections, low bytes
            df.loc[attack_mask, 'ct_srv_dst'] *= 3
            df.loc[attack_mask, 'sbytes'] *= 0.1

        elif attack == 'Exploits':
            # Exploits: Unusual port access
            df.loc[attack_mask, 'dsport'] = np.random.choice([21, 22, 23, 3389],
                                                             size=attack_mask.sum())

        elif attack == 'Fuzzers':
            # Fuzzers: High error rates
            df.loc[attack_mask, 'sloss'] *= 5
            df.loc[attack_mask, 'dloss'] *= 5

    print(f"✓ Generated {n_samples} records")
    print(f"\nAttack distribution:")
    print(df['attack_cat'].value_counts())

    return df


def save_synthetic_data(df, output_dir=None):
    """Save synthetic data to CSV files"""
    if output_dir is None:
        output_dir = Config.RAW_DATA_DIR / "UNSW-NB15"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into multiple files like the real dataset
    split_size = len(df) // 2

    df1 = df[:split_size]
    df2 = df[split_size:]

    file1 = output_dir / "UNSW-NB15_1.csv"
    file2 = output_dir / "UNSW-NB15_2.csv"

    df1.to_csv(file1, index=False, header=False)
    df2.to_csv(file2, index=False, header=False)

    print(f"\n✓ Saved synthetic data:")
    print(f"  {file1} ({len(df1)} records)")
    print(f"  {file2} ({len(df2)} records)")

    return file1, file2


def main():
    """Generate and save synthetic data"""
    print("=" * 80)
    print("Synthetic UNSW-NB15 Data Generator")
    print("=" * 80)
    print()

    # Generate data
    df = generate_synthetic_unsw_data(n_samples=50000)

    # Save data
    save_synthetic_data(df)

    print()
    print("=" * 80)
    print("Synthetic data generation complete!")
    print("You can now train the model using: python train.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
