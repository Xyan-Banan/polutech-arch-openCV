from cv2 import *
from Colors import Colors


def get_font_color(color):
    for_white = (Colors.RED, Colors.BLACK, Colors.VIOLET, Colors.GREEN, Colors.BLUE)
    # for_black = (Colors.WHITE, Colors.YELLOW)
    if color in for_white:
        return Colors.WHITE
    else:
        return Colors.BLACK


def draw_results(img, results, color, label=()):
    canvas = img.copy()
    for result in results:
        x, y, w, h = result
        rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        if label:
            scale = label[1] if len(label) > 1 else 1
            font_color = label[2] if len(label) > 2 else get_font_color(color)
            rectangle(canvas, (x, y), (int(x + 30 * scale), int(y - 10 * scale)), color, FILLED)
            putText(canvas, label[0], (x, y), FONT_HERSHEY_PLAIN, scale, font_color, 1)
    return canvas


if __name__ == "__main__":
    # classifier = CascadeClassifier("venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    # catClassifier = CascadeClassifier(haarcascades + "haarcascade_frontalcatface.xml")
    facesClassifier = CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
    catClassifier = CascadeClassifier(haarcascades + "haarcascade_frontalcatface_extended.xml")
    dogClassifier = CascadeClassifier("dog_face.xml")
    minNeighbors = 5
    scale = 1.3
    imgHeight = 250
    labelScale = 0.5

    cap = VideoCapture(0)
    success, img = cap.read()
    key = 0
    while success and key != ord('q'):
        # img = imread("images/cats_and_dogs3.jpg")
        # h, w, _ = img.shape
        # w = int(w / (h / 500))
        # img = resize(img, (w, 500))

        # flip to match camera movement
        img = flip(img, 1)
        # resize to improve classifiers speed
        h, w, _ = img.shape
        new_h = imgHeight
        new_w = int(new_h / h * w)
        img = resize(img, (new_w, new_h))

        gray = cvtColor(img, COLOR_BGR2GRAY)
        # wb = Canny(gray,th1,th2)
        # gray = wb
        # imshow("WB",wb)
        # gray = GaussianBlur(gray, (5, 5), 0)

        cats = catClassifier.detectMultiScale(gray, minNeighbors=minNeighbors, minSize=(24, 24))
        # cats2 = catClassifier2.detectMultiScale(gray, minNeighbors=minNeighbors)
        dogs = dogClassifier.detectMultiScale(gray, minNeighbors=minNeighbors, minSize=(24, 24))
        faces = facesClassifier.detectMultiScale(gray, minNeighbors=minNeighbors, minSize=(20, 20))

        # img = GaussianBlur(img, (3, 3), 1)
        img = draw_results(img, cats, Colors.YELLOW, ("cat", labelScale))
        # img = draw_results(img, cats2, Colors.BLUE, ("cat", labelScale))
        img = draw_results(img, dogs, Colors.RED, ("dog", labelScale))
        img = draw_results(img, faces, Colors.BLACK, ("human", labelScale))

        img = resize(img, (w, h))
        imshow("OpenCV", img)
        key = waitKey(1)
        if key == ord('w'):
            scale += 0.1
            print(f"scale = {scale}, minNeighbors = {minNeighbors}")
        elif key == ord('s'):
            if scale > 1.1:
                scale -= 0.1
                print(f"scale = {scale}, minNeighbors = {minNeighbors}")
        elif key == ord('d'):
            minNeighbors += 1
            print(f"scale = {scale}, minNeighbors = {minNeighbors}")
        elif key == ord('a'):
            if minNeighbors > 1:
                minNeighbors -= 1
                print(f"scale = {scale}, minNeighbors = {minNeighbors}")
        # elif key == ord('['):
        #     th1 += 10
        #     print(f"th1 = {th1} th2 = {th2}")
        # elif key == ord(';'):
        #     th1 -= 10
        #     print(f"th1 = {th1} th2 = {th2}")
        # elif key == ord(']'):
        #     th2 += 10
        #     print(f"th1 = {th1} th2 = {th2}")
        # elif key == ord("'"):
        #     th2 -= 10
        #     print(f"th1 = {th1} th2 = {th2}")

        success, img = cap.read()
        # success = False
    cap.release()
    destroyAllWindows()
