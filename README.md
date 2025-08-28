# Hand Gesture Recognition with MediaPipe + HMM

This project recognizes simple mid-air **hand-drawn letters** (`C`, `Z`, `L`) using a webcam.
It leverages **MediaPipe** to detect the **index fingertip**, smooths its motion with a **Kalman filter**, converts the trajectory into **angle bins**, and classifies the sequence using a **Hidden Markov Model (HMM)**.

---

## How to Run

### 1. Install dependencies

```bash
pip install opencv-python numpy mediapipe matplotlib scikit-learn
```

### 2. Train models

* Open the script and set:

  ```python
  MODE = "train"
  ```
* Run:

  ```bash
  python gesture_hmm.py
  ```
* Follow prompts to record samples of each gesture (`C`, `Z`, `L`).
* Trained models will be saved as `trained_models_mediapipe.pkl`.

### 3. Test gestures

* Open the script and set:

  ```python
  MODE = "test"
  ```
* Run:

  ```bash
  python gesture_hmm.py
  ```
* Use the menu:

  * **`t`** → test a gesture (draw in the air, then input the true label)
  * **`c`** → view confusion matrix + accuracy
  * **`q`** → quit
