# installed package imports
import cv2

# app module imports
from StateEstimator import StateEstimator

# Example driver code for getting state of robot at every time step
state_est = StateEstimator()
while True:
    # Print current state estimation
    x, y, vx, vy, theta, human_x, human_y, human_vx, human_vy = state_est.estimate_robot_state()
    print("----------", 
        "\nrobot x: ", x, 
        "\nrobot y: ", y,  
        "\nrobot vx: ", vx, 
        "\nrobot vy: ", vy, 
        "\nrobot theta: ", theta,
        "\nhuman x: ", human_x, 
        "\nhuman y: ", human_y, 
        "\nhuman vx: ", human_vx, 
        "\nhuman vy: ", human_vy, 
        "\n----------")

    # Quit if 'esc' or 'q' is held down
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        state_est.quit()
