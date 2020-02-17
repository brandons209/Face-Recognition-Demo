import cv2
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cam_index>")
        sys.exit(1)

    cam_index = int(sys.argv[1])

    cam = cv2.VideoCapture(cam_index)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Photo", frame)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            break
        elif k % 256 == 32:
            # SPACE pressed
            os.system("clear")
            name = input("Please enter your name:\n")
            img_name = f"{name}.png"
            img_name = os.path.join("faces", img_name)
            cv2.imwrite(img_name, frame)
            print(f"Thanks {name}! Image saved.")

    cam.release()
    cv2.destroyAllWindows()
