import cv2
import numpy as np
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt
from math import atan2, degrees
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



#come up with a confusion a matrix
#come up with the exact name of that features
#what is the hand detecting

# ------------------------------
# Configuration
# ------------------------------
LETTERS = ['C', 'Z', 'L']
ANGLE_BINS = 8
SAMPLES_PER_LETTER = 10
DATA_FILE = "gesture_data_mediapipe.pkl"
MODEL_FILE = "trained_models_mediapipe.pkl"

y_true = []
y_pred = []


# ------------------------------
# MediaPipe Setup
# ------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ------------------------------
# Kalman Filter
# ------------------------------
def init_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf

# ------------------------------
# Visualizations
# ------------------------------
def plot_trajectory(traj, title="Trajectory"):
    if not traj:
        print("Empty trajectory.")
        return
    x_vals = [p[0] for p in traj]
    y_vals = [p[1] for p in traj]

    plt.figure(figsize=(5, 5))
    plt.plot(x_vals, y_vals, marker='o')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def print_quantized_sequence(obs):
    print("Quantized sequence (angle bins):")
    print(obs)

# ------------------------------
# Gesture Capture (MediaPipe)
# ------------------------------
def capture_gesture():
    cap = cv2.VideoCapture(0)
    trajectory = []
    kf = init_kalman()
    initialized = False
    path = []
    recording = False

    print("Press 's' to start recording, 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                fingertip = hand_landmarks.landmark[8]  # Index fingertip
                x, y = int(fingertip.x * w), int(fingertip.y * h)

                measured = np.array([[np.float32(x)], [np.float32(y)]])
                if not initialized:
                    kf.statePre[:2] = measured
                    initialized = True
                kf.correct(measured)
                prediction = kf.predict()

                point = (int(prediction[0]), int(prediction[1]))

                if recording:
                    trajectory.append(point)
                    path.append(point)

                # Draw the path of the fingertip
                for i in range(1, len(path)):
                    cv2.line(frame, path[i - 1], path[i], (0, 0, 255), 2)

                # Draw a virtual object following the hand
                cv2.circle(frame, point, 20, (255, 0, 0), -1)  # Virtual object
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Actual fingertip
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Mirror Hand Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
            path = []
            print("Recording started...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return trajectory

# ------------------------------
# Quantize Trajectory
# ------------------------------
def quantize_trajectory(traj, bins=ANGLE_BINS):
    directions = []
    for i in range(1, len(traj)):
        dx = traj[i][0] - traj[i - 1][0]
        dy = traj[i][1] - traj[i - 1][1]
        angle = degrees(atan2(dy, dx)) % 360
        directions.append(angle)

    if not directions:
        return []

    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    dirs_np = np.array(directions).reshape(-1, 1)
    quantized = discretizer.fit_transform(dirs_np).astype(int).flatten()
    return quantized.tolist()

# ------------------------------
# HMM Training
# ------------------------------
def train_hmm(observations, N=6, M=ANGLE_BINS, iterations=10):
    T = len(observations)
    A = np.random.rand(N, N)
    A /= A.sum(axis=1, keepdims=True)
    B = np.random.rand(N, M)
    B /= B.sum(axis=1, keepdims=True)
    pi = np.random.rand(N)
    pi /= pi.sum()

    for _ in range(iterations):
        alpha = np.zeros((T, N))
        scale = np.zeros(T)
        alpha[0, :] = pi * B[:, observations[0]]
        scale[0] = 1.0 / (np.sum(alpha[0, :]) + 1e-12)
        alpha[0, :] *= scale[0]

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t - 1, :] * A[:, j]) * B[j, observations[t]]
            scale[t] = 1.0 / (np.sum(alpha[t, :]) + 1e-12)
            alpha[t, :] *= scale[t]

        beta = np.zeros((T, N))
        beta[-1, :] = scale[-1]
        for t in reversed(range(T - 1)):
            for i in range(N):
                beta[t, i] = np.sum(A[i, :] * B[:, observations[t + 1]] * beta[t + 1, :])
            beta[t, :] *= scale[t]

        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-12

        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denom = np.sum([
                alpha[t, i] * A[i, j] * B[j, observations[t + 1]] * beta[t + 1, j]
                for i in range(N) for j in range(N)
            ]) + 1e-12
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (
                        alpha[t, i] * A[i, j] * B[j, observations[t + 1]] * beta[t + 1, j]
                    ) / denom

        pi = gamma[0, :]
        for i in range(N):
            A[i, :] = np.sum(xi[:, i, :], axis=0) / (np.sum(gamma[:-1, i]) + 1e-12)

        for j in range(N):
            for k in range(M):
                B[j, k] = np.sum(gamma[t, j] for t in range(T) if observations[t] == k)
            B[j, :] /= (np.sum(gamma[:, j]) + 1e-12)

        A = np.nan_to_num(A / A.sum(axis=1, keepdims=True))
        B = np.nan_to_num(B / B.sum(axis=1, keepdims=True))

    return A, B, pi

# ------------------------------
# Likelihood Computation
# ------------------------------
def compute_likelihood(obs, A, B, pi):
    T, N = len(obs), len(pi)
    alpha = np.zeros((T, N))
    alpha[0, :] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1, :] * A[:, j]) * B[j, obs[t]]
    return np.sum(alpha[-1, :])

# ------------------------------
# Save/Load Data and Models
# ------------------------------
def collect_and_save_data():
    data = {letter: [] for letter in LETTERS}
    for letter in LETTERS:
        print(f"\nCollecting samples for letter '{letter}'...")
        for i in range(SAMPLES_PER_LETTER):
            print(f"Sample {i + 1}/{SAMPLES_PER_LETTER}:")
            traj = capture_gesture()
            if len(traj) < 10:
                print("Gesture too short, skipping.")
                continue
            plot_trajectory(traj, title=f"Letter {letter} - Sample {i+1}")
            obs = quantize_trajectory(traj)
            print_quantized_sequence(obs)
            if obs:
                data[letter].append(obs)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {DATA_FILE}")

def load_data_and_train_models():
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    models = {}
    for letter, sequences in data.items():
        # Flatten all sequences into one long sequence
        # OR run Baum-Welch over all separately
        all_obs = []
        for seq in sequences:
            all_obs.extend(seq)
        A, B, pi = train_hmm(all_obs)  # ✅ Use all observations
        models[letter] = (A, B, pi)
    return models


def save_models(models, filename=MODEL_FILE):
    with open(filename, 'wb') as f:
        pickle.dump(models, f)
    print(f"Trained models saved to {filename}")

def load_models(filename=MODEL_FILE):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ------------------------------
# Prediction
# ------------------------------
def test_gesture(models):
    print("\nShow a letter gesture and press 's' to start...")
    traj = capture_gesture()
    plot_trajectory(traj, title="Test Gesture")
    obs = quantize_trajectory(traj)
    print_quantized_sequence(obs)
    
    if not obs:
        print("Invalid gesture.")
        return

    scores = {}
    for letter, (A, B, pi) in models.items():
        scores[letter] = compute_likelihood(obs, A, B, pi)

    print("Likelihoods:", scores)
    prediction = max(scores, key=scores.get)
    print(f"Prediction: {prediction}")
    
    true_label = input("What it suppose to detect").strip().upper()
    if true_label in LETTERS:
        y_true.append(true_label)
        y_pred.append(prediction)
    else:
        print("Invalid label. Skipping this test.")

def show_confusion_matrix():
    if not y_true:
        print("No results to show. Run some tests first.")
        return
    cm = confusion_matrix(y_true, y_pred, labels=LETTERS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LETTERS)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix for Gesture Recognition")
    plt.show()

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}% ({acc:.2f})")



# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    MODE = "test"  # Use "train" to re-train

    if MODE == "train":
        collect_and_save_data()
        models = load_data_and_train_models()
        save_models(models)
    else:
        models = load_models()
        while True:
            choice = input("\nChoose: (t)est gesture, (c)onfusion matrix, (q)uit: ").strip().lower()
            if choice == 't':
                test_gesture(models)
            elif choice == 'c':
                show_confusion_matrix()
            elif choice == 'q':
                break
            else:
                print("Invalid choice.")

