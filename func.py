from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import tensorflow as tf
import time
import os


# comment out below line to enable tensorflow logging outputs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# deep sort imports


def tracking_func(framework='tf', weights='./checkpoints/yolov4-416', size=416, tiny=False, model='yolov4', video='./data/video/test.mp4', output=None, output_format='XVID', iou=0.45, score=0.50, dont_show=False, info=False, count=False, image_window='output_window', file1='cam1.txt', file2='cam2.txt', to_read='out.txt', jump_folder='./jumpes/1'):

    cam1 = open(file1, 'w')
    cam2 = open(file2, 'w')

    read_file = open(to_read, 'r')

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    set_coord = []
    track_dic = {}
    previous = []
    current = []
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.minor_load_config(tiny, model)
    input_size = size
    video_path = video

    # load tflite model if flag is set
    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(
            weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        # print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if count and len(read_file.read().splitlines()) > 0:

            read_file.seek(0, 0)
            color = read_file.read().splitlines()[-1]
            read_file.seek(0, 0)
            color = color.split(',')
            # print(read_file)
            # print(color)
            if color[0] == 'g':
                col = (0, 255, 0)
                cv2.putText(frame, "Time Remaining: {}".format(
                    color[1]), (800, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, col, 2)
            elif color[0] == 'r':
                col = (255, 0, 0)

                if len(color) == 2:
                    cv2.putText(frame, "Time Remaining: {}".format(
                        color[1]), (800, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, col, 2)

            cv2.putText(frame, "Objects being tracked: {}".format(
                count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, col, 2)
            # print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes

        utils.save_files(cam1, cam2, count)

        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # copy frame

        img = frame.copy()

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()

            ids = class_name + "-" + str(track.track_id)

            if ids not in track_dic.keys():
                track_dic[ids] = bbox

        # draw bbox on screen
            current = []
            current.append(class_name + "-" + str(track.track_id))
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(
                bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
                len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),
                        (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)

        # if enable info flag then print details about each track
            if info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                    str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        if len(previous) > 0:
            for prev in previous:
                if prev not in current:
                    if prev not in set_coord:

                        set_coord.append(prev)
                        box_coord = track_dic.get(prev)
                        # images = img[
                        #     int(box_coord[1]):int(box_coord[1]) + int(int(box_coord[3]-int(box_coord[1]))),
                        #     int(box_coord[0]): int(box_coord[0]) + int(int(box_coord[2]-int(box_coord[0])))]
                        images = img[
                            int(box_coord[1]):int(box_coord[3]),
                            int(box_coord[0]): int(box_coord[2])]

                        name = prev + '.jpg'
                        name = os.path.join(jump_folder, name)
                        cv2.imwrite(name, images)

        previous = current

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not dont_show:
            cv2.imshow(image_window, result)

        # if output flag is set, save video file
        if output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
