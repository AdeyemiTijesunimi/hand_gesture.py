# Hand Gesture Recognition with MediaPipe + HMM

This project recognizes simple mid-air **hand-drawn letters** (`C`, `Z`, `L`) using a webcam.
It leverages **MediaPipe** to detect the **index fingertip**, smooths its motion with a **Kalman filter**, converts the trajectory into **angle bins**, and classifies the sequence using a **Hidden Markov Model (HMM)**.

<img width="455" height="465" alt="Screenshot 2025-08-28 at 1 01 24 PM" src="https://github.com/user-attachments/assets/d889c64d-99e7-4150-8ea3-43cce2e6d5ab" />
<img width="681" height="34" alt="Screenshot 2025-08-28 at 1 02 16 PM" src="https://github.com/user-attachments/assets/b680fa58-ca4f-4074-b154-82c0fb0b9c92" />
<img width="442" height="460" alt="Screenshot 2025-08-28 at 1 05 23 PM" src="https://github.com/user-attachments/assets/7495db24-a184-47cc-a3f4-8fdd05917cfd" />
<img width="690" height="34" alt="Screenshot 2025-08-28 at 1 05 45 PM" src="https://github.com/user-attachments/assets/de10bf3f-c040-441c-b9be-aca3970948e4" />
<img width="441" height="460" alt="Screenshot 2025-08-28 at 1 07 21 PM" src="https://github.com/user-attachments/assets/d98413ee-2ef1-40fa-a422-373f3d9ec788" />
<img width="842" height="31" alt="Screenshot 2025-08-28 at 1 08 08 PM" src="https://github.com/user-attachments/assets/59c4c7c9-0299-470d-9bfa-03008cadf94e" />

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
