import numpy as np
import cv2
import pickle

from vehicle_detection import Pipeline

# Initiate The Vehicle Detection Pipeline
pipeline = Pipeline()

try:
    svc_pickle = pickle.load(open("svc_pickle.p", "rb"))
    pipeline.svc = svc_pickle["svc"]
    pipeline.X_scaler = svc_pickle["X_scaler"]
except (OSError, IOError) as e:
    pipeline.load_training_data()
    pipeline.combine_normalize_features()
    pipeline.fit_model()

    dump_p = {"svc": pipeline.svc, "X_scaler": pipeline.X_scaler}
    pickle.dump(dump_p, open("svc_pickle.p", "wb"))

name = 'project_'

cap = cv2.VideoCapture(name + 'video.mp4')

# Check if camera opened successfully
if cap.isOpened() is False:
    print("Error opening video stream or file")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.mp4' file.
out = cv2.VideoWriter(name + 'out.mp4', cv2.VideoWriter_fourcc(*'H264'), 24, (frame_width, frame_height))
i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret is True:
        if i % 2 == 0:
            frame, heatmap = pipeline.run(frame, calc=True)
        else:
            frame, heatmap = pipeline.run(frame, calc=False)

        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    i += 1

cap.release()
out.release()
cv2.destroyAllWindows()
