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
        [6.0],
        [100.0],
        [110.0],
        [120.0],
        [130.0],
    ])

    print("Original data:")
    print(X.flatten())

    # -------------------------
    # 2. First split
    # -------------------------
    X_train_val, X_test = train_test_split(
        X,
        test_size=0.2,
        random_state=42,
    )

    # -------------------------
    # 3. Second split
    # -------------------------
    X_train, X_val = train_test_split(
        X_train_val,
        test_size=0.25,
        random_state=42,
    )

    print("\n--- SPLIT ---")

    print("\nTrain:")
    print(X_train.flatten())

    print("\nValidation:")
    print(X_val.flatten())

    print("\nTest:")
    print(X_test.flatten())

    # -------------------------
    # 4. Correct preprocessing
    # -------------------------
    scaler = StandardScaler()

    # FIT ONLY ON TRAIN
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- TRAIN-ONLY FIT ---")

    print("\nScaler mean:")
    print(scaler.mean_[0])

    print("\nScaler std:")
    print(scaler.scale_[0])

    print("\nTrain scaled:")
    print(X_train_scaled.flatten())

    print("\nValidation scaled:")
    print(X_val_scaled.flatten())

    print("\nTest scaled:")
    print(X_test_scaled.flatten())

    demonstrate_wrong_preprocessing(
        X_train,
        X_val,
        X_test,
    )


def demonstrate_wrong_preprocessing(X_train, X_val, X_test):
    wrong_scaler = StandardScaler()

    X_all = np.concatenate([
        X_train,
        X_val,
        X_test,
    ])

    wrong_scaler.fit(X_all)

    print("\n--- WRONG: ALL-DATA FIT ---")
    print("Scaler mean:", wrong_scaler.mean_[0])
    print("Scaler std:", wrong_scaler.scale_[0])

if __name__ == "__main__":
    main()