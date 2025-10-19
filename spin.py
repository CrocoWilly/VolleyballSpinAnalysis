import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import queue

class BallSpinCalculator:
    def __init__(self) -> None:
        ## Find Correct Ball Parameters ##
        self.q = queue.Queue()
        self.queue_size = 5
        self.prev_frame_balls_pos = []              # frame #n-3 candidate balls info 
        self.now_frame_balls_pos = []               # frame #n candidate balls info
        self.distance_high_threshold = 150
        self.distance_low_threshold = 10
        self.prev_draw_box = []


        ## Optical Flow Parameters ##
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)                                    # ShiTomasi corner detection parameters
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))   # Parameters for Lucas-Kanade optical flow
    
        self.old_gray = []
        self.prev_points = []

        self.frame_interval = 3                     # every 3 frame, calculate the spin rate and find a new set of optical flow points
        self.old_spin_rate = 0


    # Set next iteration of the parameters for finding the correct ball 
    def set_next_iteration(self):
        self.q.put(self.now_frame_balls_pos)
        if(self.q.qsize() > self.queue_size):
            self.prev_frame_balls_pos = self.q.get()

        self.now_frame_balls_pos = []


    # Record now frame boxes info, for the next step calculation
    def extract_and_assign_now_boxes(self, boxes):
        for xywh, class_id, conf in zip(boxes.xywh, boxes.cls, boxes.conf):
            x, y, w, h = xywh.tolist()

            # append ball center representing a ball object
            self.now_frame_balls_pos.append((int(x), int(y)))


    # Calculate Euclidean distance between the center of two boxes
    def calculate_distance(self, ball1_center, ball2_center):
        distance = np.sqrt((ball1_center[0] - ball2_center[0])**2 + (ball1_center[1] - ball2_center[1])**2)
        return distance

    # The Hungarian algorithm
    def track_objects(self):
        ## Number of boxes in each frame
        num_balls_prev = len(self.prev_frame_balls_pos)
        num_balls_now = len(self.now_frame_balls_pos)
        
        ## Create a cost matrix from distances between boxes in consecutive frames
        cost_matrix = np.zeros((num_balls_prev, num_balls_now))

        cost_matrix_penalty = 100000
        
        for i, prev_ball_pos in enumerate(self.prev_frame_balls_pos):
            for j, now_ball_pos in enumerate(self.now_frame_balls_pos):

                ## exclude the unmatched box when prev_frame box nums is not equal to now_frame box nums 
                distance = self.calculate_distance(prev_ball_pos, now_ball_pos)
                if distance > self.distance_high_threshold:              # Apply the distance threshold
                    cost_matrix[i, j] = cost_matrix_penalty              # Assign a high cost to unacceptable pairs
                else:
                    cost_matrix[i, j] = distance

        
        ## Apply the Hungarian algorithm (linear sum assignment) to find the minimum cost assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        ## Output the results
        results = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] != cost_matrix_penalty:
                results.append((r, c))
        return results


    # Apply the Hungarian algorithm to find correct ball
    def find_correct_ball(self, boxes):

        ## Assign each corresponding ball by Hungarian algorithm 
        self.extract_and_assign_now_boxes(boxes)

        ## Assign candidate draw boxes
        candidate_draw_boxes = []
        if self.prev_frame_balls_pos:       # when prev_frame_balls_pos != NULL
            balls_asignment_results = self.track_objects()

            for index, (old_index, new_index) in enumerate(balls_asignment_results):
                old_box = self.prev_frame_balls_pos[old_index]
                new_box = self.now_frame_balls_pos[new_index]

                # filter out the ball outside the game court
                if(abs(new_box[0] - old_box[0]) + abs(new_box[1] - old_box[1]) >= self.distance_low_threshold):        # Manhattan Distance < threshold -> regard as static
                    candidate_draw_boxes.append(new_box)

        ## Find the correct ball
        if self.prev_draw_box and candidate_draw_boxes:
            min_distance = 0
            min_index = 0
            for i in range(len(candidate_draw_boxes)):
                if(self.calculate_distance(self.prev_draw_box, candidate_draw_boxes[i]) < min_distance):
                    min_index = i

            self.prev_draw_box = candidate_draw_boxes[min_index]
            return candidate_draw_boxes[min_index]
        else:
            if candidate_draw_boxes:
                self.prev_draw_box = candidate_draw_boxes[0]
        
        return []
        

    # Calculate the spin by Phy-OptiCoord method designed by Yen-Chang Chen 
    def calculate_spin_rate(self, prev_points, curr_points, frame_time):
        
        ## STEP1. Regard the 100*100 bounding box as a coordinate system ##
        # (i.e., ball center = O(0, 0), and each detected optical point = (x, y) in the coordinate system)
        ball_center = [50, 50]
        # Compute the coordinate of each point relative to the ball's center -> regard ball center as (0, 0)
        coordinates_prev = prev_points - ball_center
        coordinates_curr = curr_points - ball_center


        ## STEP2. Use arctan(y/x) to find the theta of each point on the coordinate system ##
        angles_prev = np.arctan2(coordinates_prev[:, 1], coordinates_prev[:, 0])
        angles_curr = np.arctan2(coordinates_curr[:, 1], coordinates_curr[:, 0])


        ## STEP3. Calculate the rotation amount of each point from frame #n-3 to frame #n by using "the predefined quadrant rotational rules" ##
        rotations_quantity = []
        for i in range(len(coordinates_prev)):
            coord_prev = coordinates_prev[i]
            coord_curr = coordinates_prev[i]

            rotation_quantity = 0

            ##    The predefined quadrant rotational rules    ##
            # ---------  同象限內分析: 0rpm ~ 300rpm --------- #
            # case 1: 同象限內轉        theta = abs(cur - prev)
            if( (coord_prev[0] > 0 and coord_prev[1] > 0 and coord_curr[0] > 0 and coord_curr[1] > 0) or 
                (coord_prev[0] < 0 and coord_prev[1] > 0 and coord_curr[0] < 0 and coord_curr[1] > 0) or
                (coord_prev[0] < 0 and coord_prev[1] < 0 and coord_curr[0] < 0 and coord_curr[1] < 0) or
                (coord_prev[0] > 0 and coord_prev[1] < 0 and coord_curr[0] > 0 and coord_curr[1] < 0) ):
                rotation_quantity = abs(angles_curr[i] - angles_prev[i])
            
            # ---------  跨一象限分析: 0rpm ~ 600rpm --------- #
            # case 2: 一四(四一)象限轉  theta = abs(cur - prev)
            elif( (coord_prev[0] > 0 and coord_prev[1] > 0 and coord_curr[0] > 0 and coord_curr[1] < 0) or 
                  (coord_prev[0] > 0 and coord_prev[1] < 0 and coord_curr[0] > 0 and coord_curr[1] > 0) ):
                rotation_quantity = abs(angles_curr[i] - angles_prev[i])

            # case 3: 三四(四三)象限轉  theta = 180度 - abs(cur - prev)
            elif( (coord_prev[0] > 0 and coord_prev[1] < 0 and coord_curr[0] < 0 and coord_curr[1] < 0) or 
                  (coord_prev[0] < 0 and coord_prev[1] < 0 and coord_curr[0] > 0 and coord_curr[1] < 0) ):
                rotation_quantity = np.pi - abs(angles_curr[i] - angles_prev[i])

            # case 4: 二三(三二)象限轉  theta = abs(cur - prev)
            elif( (coord_prev[0] < 0 and coord_prev[1] > 0 and coord_curr[0] < 0 and coord_curr[1] < 0) or 
                  (coord_prev[0] < 0 and coord_prev[1] < 0 and coord_curr[0] < 0 and coord_curr[1] > 0) ):
                rotation_quantity = abs(angles_curr[i] - angles_prev[i])

            # case 5: 一二象限轉        theta = 180度 - abs(cur - prev)
            elif( (coord_prev[0] < 0 and coord_prev[1] > 0 and coord_curr[0] > 0 and coord_curr[1] > 0) or 
                  (coord_prev[0] > 0 and coord_prev[1] > 0 and coord_curr[0] < 0 and coord_curr[1] > 0) ):
                rotation_quantity = np.pi - abs(angles_curr[i] - angles_prev[i])
            
            # ---------  跨兩象限分析(很少情況會發生): 300rpm ~ 900rpm --------- #
            # case 6: 順一三、二四轉    theta = 180度 - (cur - prev)
            # case 7: 逆三一、四二轉    theta = 180度 + (cur - prev)
            elif( (coord_prev[0] > 0 and coord_prev[1] > 0 and coord_curr[0] < 0 and coord_curr[1] < 0) or 
                  (coord_prev[0] < 0 and coord_prev[1] > 0 and coord_curr[0] > 0 and coord_curr[1] < 0) or
                  (coord_prev[0] < 0 and coord_prev[1] < 0 and coord_curr[0] > 0 and coord_curr[1] > 0) or
                  (coord_prev[0] > 0 and coord_prev[1] < 0 and coord_curr[0] < 0 and coord_curr[1] > 0) ):
                clockwise_rotation_quantity = np.pi - (angles_curr[i] - angles_prev[i])
                counterclockwise_rotation_quantity = np.pi + (angles_curr[i] - angles_prev[i])
                rotation_quantity = min(clockwise_rotation_quantity, counterclockwise_rotation_quantity)

            # case 8: 如果有點在x, y軸上，不計此點
            else:
                rotation_quantity = 0

            rotations_quantity.append(round(rotation_quantity, 5))

        ## STEP4. Take the median of the n rotation amounts and calculate the angular velocity in RPM  ##
        if len(rotations_quantity) != 0:
            avg_rotation_quantity = np.median(rotations_quantity)
        else:
            avg_rotation_quantity = 0

        angular_velocity = avg_rotation_quantity / frame_time
        spin_rate_rpm = (angular_velocity * 60) / (2 * np.pi)
        
        return spin_rate_rpm


    # Find Optical flow points and return calculated spin 
    def find_points_and_calculate_spin(self, frame, frame_no):
    
        analysis_roi = frame[0:100, 0:100]

        if len(self.old_gray) == 0 or frame_no % self.frame_interval == 1:                    # 如果還沒有上一偵資訊，就先不要算LK algorithm，先assign好old資訊
            ## Convert ROI to grayscale
            self.old_gray = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2GRAY)

            ## Key points detection: use Shi-Tomashi method Detect corners to track
            self.prev_points = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

        else:
            ## Convert new ROI to grayscale
            new_gray = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2GRAY)

            ## Calculate optical flow using Lucas-Kanade method
            cur_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, new_gray, self.prev_points, None, **self.lk_params)

            ## Select good points
            if cur_points is not None:
                valid_cur_points = cur_points[st == 1]
                valid_prev_points = self.prev_points[st == 1]

                ## Draw the tracks of optical flow on the screen
                for i, (new, old) in enumerate(zip(valid_cur_points, valid_prev_points)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)


                ## Calculate Spin every 3 frames
                if(frame_no % self.frame_interval == 0):
                    new_spin_rate = self.calculate_spin_rate(valid_prev_points, valid_cur_points, 1/60)
                
                    ## STEP6. Judge if the calculated spin rate is reasonable and smoothing the calculated RPM value ##
                    if not (np.isnan(new_spin_rate) or new_spin_rate <= 0 or new_spin_rate > 600):
                        self.old_spin_rate = int(0.8 * self.old_spin_rate + 0.2 * new_spin_rate)

                ## Update the previous frame and previous points
                self.prev_points = valid_cur_points.reshape(-1, 1, 2)
                
            self.old_gray = new_gray.copy()
            
        return frame, self.old_spin_rate


    