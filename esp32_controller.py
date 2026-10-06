import cv2
import math
import numpy as np
import mediapipe as mp
import time
import serial


# ESP32 serial settings
ESP32_PORT = "COM5"
ESP32_BAUD = 115200

try:
    esp32 = serial.Serial(
        ESP32_PORT,
        ESP32_BAUD,
        timeout=1
    )

    time.sleep(2)
    print("ESP32 connected:", ESP32_PORT)

except Exception as e:
    print("ESP32 connection failed:", e)
    esp32 = None


# MediaPipe hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# Camera and display window
cap = cv2.VideoCapture(0)

cv2.namedWindow(
    "Curtain Puppet ESP32 Version",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Curtain Puppet ESP32 Version",
    1280,
    800
)


# Curtain settings
curtain_progress = 0.0
is_opened = False

CURTAIN_SPEED = 0.009
THUMB_OPEN_THRESHOLD = 0.18


# Puppet angle
puppet_angle = 0


# Volume control
volume_level = 1
volume_locked = False

last_volume = 1
last_change_time = time.time()

HOLD_TIME = 3.0

left_was_detected = False


# Send data to ESP32 about 20 times per second
last_send_time = 0
SEND_INTERVAL = 0.05


def distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def draw_puppet(img, center_x, base_y, angle):

    angle = max(-40, min(40, angle))

    rad = math.radians(angle)

    pivot_x = center_x
    pivot_y = base_y

    body_len = 150
    body_width = 70
    head_offset = 190

    # Draw the puppet body
    top_x = int(
        pivot_x +
        math.sin(rad) * body_len
    )

    top_y = int(
        pivot_y -
        math.cos(rad) * body_len
    )

    side_x = int(
        math.cos(rad) *
        (body_width / 2)
    )

    side_y = int(
        math.sin(rad) *
        (body_width / 2)
    )

    body_pts = np.array(
        [
            [
                top_x - side_x,
                top_y - side_y
            ],
            [
                top_x + side_x,
                top_y + side_y
            ],
            [
                pivot_x + side_x,
                pivot_y + side_y
            ],
            [
                pivot_x - side_x,
                pivot_y - side_y
            ]
        ],
        dtype=np.int32
    )

    cv2.fillPoly(
        img,
        [body_pts],
        (80, 140, 220)
    )

    # Draw the head
    head_x = int(
        pivot_x +
        math.sin(rad) * head_offset
    )

    head_y = int(
        pivot_y -
        math.cos(rad) * head_offset
    )

    cv2.circle(
        img,
        (head_x, head_y),
        35,
        (220, 190, 140),
        -1
    )

    # Eyes
    eye_dx = int(
        math.cos(rad) * 12
    )

    eye_dy = int(
        math.sin(rad) * 12
    )

    cv2.circle(
        img,
        (
            head_x - eye_dx,
            head_y - eye_dy
        ),
        4,
        (0, 0, 0),
        -1
    )

    cv2.circle(
        img,
        (
            head_x + eye_dx,
            head_y + eye_dy
        ),
        4,
        (0, 0, 0),
        -1
    )

    # Mouth
    cv2.ellipse(
        img,
        (
            head_x,
            head_y + 12
        ),
        (12, 6),
        angle,
        0,
        180,
        (0, 0, 0),
        2
    )

    # Arms
    arm_x = int(
        pivot_x +
        math.sin(rad) * 95
    )

    arm_y = int(
        pivot_y -
        math.cos(rad) * 95
    )

    left_shoulder = (
        arm_x - side_x,
        arm_y - side_y
    )

    right_shoulder = (
        arm_x + side_x,
        arm_y + side_y
    )

    cv2.line(
        img,
        left_shoulder,
        (
            left_shoulder[0] - 70,
            left_shoulder[1] + 30
        ),
        (220, 190, 140),
        8
    )

    cv2.line(
        img,
        right_shoulder,
        (
            right_shoulder[0] + 70,
            right_shoulder[1] + 30
        ),
        (220, 190, 140),
        8
    )

    # Center point of the puppet
    cv2.circle(
        img,
        (pivot_x, pivot_y),
        12,
        (255, 255, 255),
        -1
    )


def send_to_esp32(angle, volume):

    global last_send_time

    if esp32 is None:
        return

    current_time = time.time()

    if current_time - last_send_time < SEND_INTERVAL:
        return

    last_send_time = current_time

    angle = max(-40, min(40, angle))
    volume = max(1, min(10, volume))

    command = (
        f"ANGLE:{angle},"
        f"VOLUME:{volume}\n"
    )

    try:
        esp32.write(
            command.encode("utf-8")
        )

        print(
            "ESP32 TX:",
            command.strip()
        )

    except Exception as e:
        print(
            "ESP32 Serial Error:",
            e
        )


# Main loop
while cap.isOpened():

    success, img = cap.read()

    if not success:
        print("Camera read failed")
        break

    # Flip the camera image like a mirror
    img = cv2.flip(img, 1)

    h, w, _ = img.shape


    # Process the image with MediaPipe
    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)


    right_hand_x = None
    left_detected = False


    # Check if any hands are detected
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


        # Use the distance between two thumbs to open the curtain
        if len(result.multi_hand_landmarks) >= 2:

            hand1 = result.multi_hand_landmarks[0]
            hand2 = result.multi_hand_landmarks[1]

            thumb1 = hand1.landmark[4]
            thumb2 = hand2.landmark[4]

            thumb_distance = distance(
                thumb1,
                thumb2
            )

            if thumb_distance > THUMB_OPEN_THRESHOLD:
                is_opened = True


        # Check left and right hands
        if result.multi_handedness:

            for index, handedness in enumerate(
                result.multi_handedness
            ):

                label = (
                    handedness
                    .classification[0]
                    .label
                )

                hand = (
                    result
                    .multi_hand_landmarks[index]
                )


                # Right hand controls the puppet
                if label == "Right":

                    wrist = hand.landmark[0]
                    right_hand_x = wrist.x


                # Left hand controls the volume
                if (
                    label == "Left"
                    and curtain_progress >= 1.0
                ):

                    left_detected = True

                    left_thumb = hand.landmark[4]
                    left_index = hand.landmark[8]

                    left_dist = distance(
                        left_thumb,
                        left_index
                    )

                    new_volume = int(
                        (left_dist - 0.03)
                        / 0.22
                        * 10
                    )

                    new_volume = max(
                        1,
                        min(
                            10,
                            new_volume
                        )
                    )


                    # Unlock the volume when the hand appears again
                    if not left_was_detected:

                        volume_locked = False
                        last_volume = new_volume
                        last_change_time = time.time()


                    # The volume needs to stay still for a while
                    if not volume_locked:

                        volume_level = new_volume

                        if new_volume != last_volume:

                            last_volume = new_volume
                            last_change_time = time.time()


                        if (
                            time.time()
                            - last_change_time
                            >= HOLD_TIME
                        ):

                            volume_locked = True

                            print(
                                "Volume Locked:",
                                volume_level
                            )


    left_was_detected = left_detected


    # Open the curtain little by little
    if is_opened:
        curtain_progress += CURTAIN_SPEED

    curtain_progress = min(
        1.0,
        curtain_progress
    )


    # Create the stage overlay
    overlay = img.copy()

    curtain_color = (
        80,
        0,
        120
    )

    closed_width = w // 2

    open_width = int(
        w * 0.18
    )

    current_width = int(
        closed_width -
        (
            closed_width
            - open_width
        )
        * curtain_progress
    )


    # Dark stage area
    cv2.rectangle(
        overlay,
        (
            current_width,
            0
        ),
        (
            w - current_width,
            h
        ),
        (
            30,
            30,
            30
        ),
        -1
    )


    # Start controlling the puppet after the curtain is open
    if curtain_progress >= 1.0:

        if right_hand_x is not None:

            puppet_angle = int(
                (right_hand_x - 0.5)
                * 140
            )

            puppet_angle = max(
                -40,
                min(
                    40,
                    puppet_angle
                )
            )


        draw_puppet(
            overlay,
            w // 2,
            int(h * 0.82),
            puppet_angle
        )


        # Send the current values to ESP32
        send_to_esp32(
            puppet_angle,
            volume_level
        )


    # Left curtain
    cv2.rectangle(
        overlay,
        (0, 0),
        (
            current_width,
            h
        ),
        curtain_color,
        -1
    )


    # Right curtain
    cv2.rectangle(
        overlay,
        (
            w - current_width,
            0
        ),
        (
            w,
            h
        ),
        curtain_color,
        -1
    )


    # Combine the camera and stage
    img = cv2.addWeighted(
        overlay,
        0.55,
        img,
        0.45,
        0
    )


    # Show volume and puppet status
    if curtain_progress >= 1.0:

        cv2.putText(
            img,
            f"volume {volume_level}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            f"angle {puppet_angle}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )

        if volume_locked:

            cv2.putText(
                img,
                "LOCK",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255
                ),
                1,
                cv2.LINE_AA
            )


    # Show the final result
    cv2.imshow(
        "Curtain Puppet ESP32 Version",
        img
    )


    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Clean up
cap.release()
hands.close()

if esp32 is not None:

    if esp32.is_open:
        esp32.close()

cv2.destroyAllWindows()

print("Program closed")
