import torch
import numpy as np
import cv2
# import pafy
import time
import hubconf
import first_frame


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """

    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'mps'
        print("\n\nDevice Used:", self.device)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        # path = "./yolov5"
        model = torch.hub.load("ultralytics/yolov5", 'custom',
                               "/Users/illmonk/Documents/python_scripts/blackjack-bot/yolov5/runs/train/exp44/weights/best.pt")
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                print(self.class_to_label(labels[i]))
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] *
                                                                                   x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def one_frame(self):
        '''This function will check one frame.'''
        frame = cv2.imread("../assets/71.png")

        position_dict = first_frame.main(frame)

        card_slot_frame = frame[position_dict["card_slot"]
                                [0][1]:position_dict["card_slot"][1][1], position_dict["card_slot"]
                                [0][0]:position_dict["card_slot"][1][0]]

        left_side = card_slot_frame[:card_slot_frame.shape[0],
                                    :card_slot_frame.shape[1] // 2]
        right_side = card_slot_frame[:card_slot_frame.shape[0],
                                     card_slot_frame.shape[1] // 2:]

        split = np.zeros((64, 64, 3), np.uint8)
        # print(split.shape)

        h = round((64 - left_side.shape[0]) // 2)
        w = ((64 - (left_side.shape[1] + right_side.shape[1])) // 2)
        print(w, h)

        right_side = cv2.rotate(right_side, cv2.ROTATE_180)
        split[h: h + left_side.shape[0], w:w + left_side.shape[1]] = left_side
        # print(left_side.shape)

        split[h:h + left_side.shape[0], left_side.shape[1] + w:w +
              left_side.shape[1] + right_side.shape[1]] = right_side
        gray_split = cv2.cvtColor(split, cv2.COLOR_BGR2GRAY)

        # split = cv2.resize(split, (0, 0), fx=8, fy=8)

        # cv2.putText(split, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow("split", gray_split)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        results = self.score_frame(gray_split)
        print(results)
        split = self.plot_boxes(results, split)
        cv2.imshow("split", split)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = cv2.VideoCapture("../../../../Desktop/wamp.mov")

        count = 0
        img_count = 0

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            if not ret:
                break

            if count == 0:
                position_dict = first_frame.main(frame)
                # print(position_dict)
            else:
                # print(count)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(frame, (position_dict['top-left']),
                              (position_dict['bottom_right']), (0, 255, 0), 4)
                frame = cv2.putText(
                    frame, 'Game', (position_dict['top-left'][0], position_dict['top-left'][1] - 10), font, .5, (0, 255, 0), 1, cv2.LINE_AA)

                # first_frame.get_boxes(frame, (position_dict['top-left']),
                #                       (position_dict['bottom_right']))
                back_width = 0
                back_height = 10
                card_slot_frame = frame[position_dict["card_slot"]
                                        [0][1] - back_width:position_dict["card_slot"][1][1] + back_width, position_dict["card_slot"]
                                        [0][0] - back_height:position_dict["card_slot"][1][0] + back_height]

                img_count = img_count + 1
                left_side = card_slot_frame[:card_slot_frame.shape[0],
                                            :card_slot_frame.shape[1] // 2]
                right_side = card_slot_frame[card_slot_frame.shape[0] // 2:,
                                             card_slot_frame.shape[1] // 2:]

                split = np.zeros((left_side.shape[0], left_side.shape[1] + right_side.shape[1], 3), np.uint8)
                # print(split.shape)

                right_side = cv2.rotate(right_side, cv2.ROTATE_180)

                split[:left_side.shape[0], :left_side.shape[1]] = left_side
                split[:(card_slot_frame.shape[0]//2) + 1, left_side.shape[1]: left_side.shape[1] + right_side.shape[1]] = right_side
                
                gray_split = cv2.cvtColor(split, cv2.COLOR_BGR2GRAY)

                # split = cv2.resize(split, (0, 0), fx=8, fy=8)

                # cv2.putText(split, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                # gray_split = cv2.resize(gray_split, (0, 0), fx=1.163, fy=1.163)
                # print(gray_split.shape)
                results = self.score_frame(gray_split)
                split = self.plot_boxes(results, split)
                # print(results)
                # print(results)
                split = cv2.resize(split, (0, 0), fx=4, fy=4)
                cv2.imshow("split", gray_split)

            count = count + 1

            # cv2.imshow("img", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Create a new object and execute.
detection = ObjectDetection()
detection()
