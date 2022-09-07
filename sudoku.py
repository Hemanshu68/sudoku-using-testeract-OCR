# imports
import cv2
import numpy as np
from board import Board
from PIL import Image
import pytesseract


class Sudoku:
    TES_CONFIG = "--psm 10 -c tessedit_char_whitelist=123456789"
    WAIT_TIME = 0
    MAX_HEIGHT = int(1000)
    file_name = ""

    def __init__(self, file_name) -> None:
        self.file_name = file_name

    def read_file(self, filename: str) -> np.ndarray:
        return cv2.imread(filename)

    def resize(self, img: np.ndarray, max_height=MAX_HEIGHT) -> np.ndarray:
        if len(img) > max_height:
            new_height = max_height
            new_width = len(img[0]) / (len(img) / max_height)
            return cv2.resize(img, (int(new_height), int(new_width)))
        return img

    def show(self, image: np.ndarray, text: str) -> None:
        """
        Shows the image
        """
        img = image.copy()
        img = self.resize(img=img, max_height=500)
        cv2.imshow(text, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def remove_rows(self, img: np.ndarray) -> np.ndarray:
        """
        Removes horizontal lines from image
        """
        img2 = img.copy()
        rows_to_del = []

        for i, row in enumerate(img2):
            if max(row) == min(row):
                rows_to_del.append(i)

        for i in reversed(rows_to_del):
            img2 = np.delete(img2, i, 0)

        return img2

    def remove_cols(self, img: np.ndarray) -> np.ndarray:
        """
        Removes vertical lines from image
        """
        return np.transpose(self.remove_rows(np.transpose(img)))

    def read_digit(self, img) -> float:
        """
        Converts image of a number to float
        """
        pil = Image.fromarray(img.astype("uint8"))
        num = pytesseract.image_to_string(pil, config=self.TES_CONFIG)
        if num == "":
            return 0.0
        return float(num)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # apply threshold/morph
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # remove row and column lines
        img = self.remove_cols(img)
        return self.remove_rows(img)

    def define_digit_boxes(self, img, display=False) -> list:
        cnts, _ = cv2.findContours(
            img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for c in cnts:
            box = cv2.boundingRect(c)
            if 500 > box[2] > 5 and 500 > box[3] > 5:
                boxes.append(box)

        if display:
            img2 = img.copy()
            for (x, y, w, h) in boxes:
                cx = x / (len(img) / 9)
                cy = y / (len(img[0]) / 9)
                print(cx, cy)
                string = "({}, {})".format(int(cx), int(cy))
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    img2,
                    string,
                    (x + w - 60, y + h + 20),
                    color=(255, 0, 0),
                    fontFace=1,
                    fontScale=1,
                )
            self.show(img2, text="Mapped Image")

        return boxes

    def populate_board(self, img: np.ndarray, boxes: list) -> list[list[int]]:
        """
        Populates board with OCR'd numbers from img
        """
        board = np.zeros((9, 9))
        for (x, y, w, h) in boxes:
            cx = x / (len(img) / 9)
            cy = y / (len(img[0]) / 9)
            cx, cy = int(cx), int(cy)
            board[cx][cy] = self.read_digit(
                img[y - 10 : y + h + 10, x - 10 : x + w + 10]
            )

        board = np.transpose(board).tolist()
        return [[int(x) for x in row] for row in board]

    def paint_answers(self, img, answers):
        img = img.copy()
        factor = len(img) / 9
        for x, row in enumerate(answers):
            for y, val in enumerate(row):
                if val > 0:
                    cv2.putText(
                        img,
                        str(val),
                        (int((y + 0.25) * factor), int((x + 0.8) * factor)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        color=(144, 5, 250),
                        thickness=3,
                        fontScale=3,
                        lineType=cv2.LINE_AA,
                    )

        return img

    def solve(self):
        # read and resize image
        img = self.read_file(filename=self.file_name)
        img = self.resize(img)

        # copy and show image
        original = img.copy()
        # self.show(original, "Original Image")

        # process the image
        preprocessed = self.preprocess(img)
        boxes = self.define_digit_boxes(preprocessed)

        # solve the sudoku
        board = self.populate_board(preprocessed, boxes)
        answers = Board(board).solve()

        # put solution to image and show image
        result_img = self.paint_answers(original, answers)
        # self.show(result_img, "OUTPUT")
        cv2.imwrite("output.png", result_img)
