"""PASTE YOUR TEAMMATE'S CODE HERE. Leave READY=False until it works.

While READY=False, the web app uses pretrained MediaPipe in webcam mode.
When you set READY=True and restart Streamlit, a new detector choice,
"Person 2 model (webcam)", appears alongside "MediaPipe pretrained".
The game logic, prompts, webcam bridge, and scoring rules remain unchanged.

This module is the ONLY file you need to change for the model handoff.

Implement predict_expression(frame_bgr, timestamp) below. The input is an
OpenCV BGR numpy array and a monotonic timestamp. Return a dict like:

    {
        "face_detected": True,
        "smile_score": 0.87,  # score 0..1, higher = stronger smile
        "timestamp": timestamp,
        "bbox": (x, y, width, height),  # optional, in ORIGINAL frame pixels
    }

The model can work on a resized/cropped copy internally, but must map its
bbox back to original-frame coordinates. It must not open the webcam itself.
"""
# STEP 1: Paste any imports, pretrained weights loading, and helper functions
#         provided by Person 2 above this line.
# STEP 2: Implement predict_expression() below and verify it returns a dict.
# STEP 3: Change this switch to True, restart Streamlit, and choose
#         "Person 2 model (webcam)" from the Detector dropdown.
READY = False


def predict_expression(frame_bgr, timestamp: float):
    # REPLACE THE BODY OF THIS FUNCTION WITH YOUR PEER'S REAL PREDICTIONS.
    # The examples above document the required return keys.
    raise NotImplementedError("Paste Person 2's inference code here")
