import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # -------------------------
    # 1. Create synthetic data
    # -------------------------
    np.random.seed(42)

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [100.0],
        [110.0],
        [120.0],
    ])

    # -------------------------
    # 2. Split first
    # -------------------------
    X_train, X_val = train_test_split(
        X,
        test_size=0.25,
        random_state=42,
    )

    print("X_train:")
    print(X_train.flatten())

    print("\nX_val:")
    print(X_val.flatten())

    # -------------------------
    # 3. WRONG: fit on all data
    # -------------------------
    wrong_scaler = StandardScaler()

    X_all = np.concatenate([X_train, X_val])

    wrong_scaler.fit(X_all)

    print("\n--- WRONG SCALER ---")
    print("Mean:", wrong_scaler.mean_[0])
    print("Std:", wrong_scaler.scale_[0])

    # -------------------------
    # 4. CORRECT: fit on train only
    # -------------------------
    correct_scaler = StandardScaler()

    correct_scaler.fit(X_train)

    print("\n--- CORRECT SCALER ---")
    print("Mean:", correct_scaler.mean_[0])
    print("Std:", correct_scaler.scale_[0])


if __name__ == "__main__":
    main()