from vehicle_detection import Pipeline
import glob
import cv2
import matplotlib.pyplot as plt
import pickle

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

test_images = glob.glob('test_images/*.jpg')
for test_img_path in test_images:
    test_img = cv2.imread(test_img_path)

    draw_img, heatmap = pipeline.run(test_img, clear_boxes=True, thresh=1)

    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    plt.imshow(draw_img)
    plt.show()
    plt.imshow(heatmap, cmap="hot")
    plt.show()