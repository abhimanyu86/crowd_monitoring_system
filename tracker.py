import cv2
import math
import time

class PeopleTracker:
    """
    A class to track people and count entries/exits based on defined lanes and thresholds.
    Supports 'Eagle‑Eye' mode (total count in view) and 'Lane Counter' mode (entry/exit counts).
    """
    def __init__(self, mode='Eagle‑Eye'):
        """
        Initializes the PeopleTracker.
        Args:
            mode (str): The tracking mode ('Eagle‑Eye' or 'Lane Counter').
        """
        self.mode = mode
        self.lane_split_x = None # X-coordinate for the lane splitting line
        self.entry_y_threshold = None # Y-coordinate for the entry line (bottom of frame)
        self.exit_y_threshold = None  # Y-coordinate for the exit line (top of frame)

        # Tracking variables for Lane Counter mode
        self.tracked_objects = {} # Stores active tracked persons: {id: {'bbox': ..., 'centroid': ..., 'lane': 'enter'/'exit', 'counted_in': False, 'counted_out': False, 'last_frame_id': ...}}
        self.next_id = 0 # Counter for assigning unique IDs to new tracks
        self.in_count = 0 # Total count of people who entered
        self.out_count = 0 # Total count of people who exited
        self.frame_id = 0 # Current frame number for managing stale tracks

        # Configuration for tracking (adjust as needed)
        self.max_dist_sq = 5000 # Maximum squared distance for centroid matching (pixels)
        self.stale_frame_threshold = 30 # How many frames before a track is considered stale and removed

    def _get_centroid(self, bbox):
        """Calculates the centroid of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _distance_sq(self, p1, p2):
        """Calculates the squared Euclidean distance between two points."""
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def reset_counts(self):
        """Resets the entry and exit counts."""
        self.in_count = 0
        self.out_count = 0
        self.tracked_objects = {} # Clear all tracked objects on reset
        self.next_id = 0
        self.frame_id = 0

    def update(self, detections, frame):
        """
        Updates the frame with detections, performs tracking, and calculates counts.
        Args:
            detections (list): A list of tuples, where each tuple contains
                               (x1, y1, x2, y2, label) for a detected object.
            frame (numpy.ndarray): The current video frame.
        Returns:
            tuple: A tuple containing in_count, out_count, net_people_total, and the annotated frame.
        """
        self.frame_id += 1
        h, w, _ = frame.shape

        # Initialize thresholds and lane split if not already set
        if self.mode == 'Lane Counter':
            if self.lane_split_x is None:
                self.lane_split_x = w // 2
            if self.entry_y_threshold is None:
                self.entry_y_threshold = int(h * 0.9) # 90% from top (bottom of frame)
            if self.exit_y_threshold is None:
                self.exit_y_threshold = int(h * 0.1) # 10% from top (top of frame)

            # Draw the lane splitting line
            cv2.line(frame, (self.lane_split_x, 0), (self.lane_split_x, h), (0, 255, 255), 2)
            # Draw entry/exit threshold lines
            cv2.line(frame, (0, self.entry_y_threshold), (w, self.entry_y_threshold), (0, 255, 0), 1) # Green for entry
            cv2.line(frame, (0, self.exit_y_threshold), (w, self.exit_y_threshold), (0, 0, 255), 1) # Red for exit
            cv2.putText(frame, 'Entry Line', (10, self.entry_y_threshold - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, 'Exit Line', (10, self.exit_y_threshold + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


        current_persons_detections = [d for d in detections if d[4] == 'person']
        updated_tracked_ids = set()

        # --- Update existing tracks and add new ones ---
        for det_x1, det_y1, det_x2, det_y2, label in current_persons_detections:
            det_bbox = (det_x1, det_y1, det_x2, det_y2)
            det_centroid = self._get_centroid(det_bbox)
            
            matched_id = None
            min_dist = float('inf')

            # Try to match current detection with existing tracked objects
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data['status'] == 'active': # Only consider active tracks
                    dist = self._distance_sq(det_centroid, obj_data['centroid'])
                    if dist < min_dist and dist < self.max_dist_sq:
                        min_dist = dist
                        matched_id = obj_id
            
            if matched_id is not None:
                # Update existing track
                self.tracked_objects[matched_id]['bbox'] = det_bbox
                self.tracked_objects[matched_id]['centroid'] = det_centroid
                self.tracked_objects[matched_id]['last_frame_id'] = self.frame_id
                updated_tracked_ids.add(matched_id)
            else:
                # Create a new track
                self.tracked_objects[self.next_id] = {
                    'bbox': det_bbox,
                    'centroid': det_centroid,
                    'lane': 'unknown', # Will be determined below
                    'status': 'active',
                    'counted_in': False,
                    'counted_out': False,
                    'last_frame_id': self.frame_id
                }
                updated_tracked_ids.add(self.next_id)
                self.next_id += 1

        # --- Remove stale tracks ---
        stale_ids = [
            obj_id for obj_id, obj_data in self.tracked_objects.items()
            if self.frame_id - obj_data['last_frame_id'] > self.stale_frame_threshold
        ]
        for obj_id in stale_ids:
            del self.tracked_objects[obj_id]

        # --- Counting and Drawing ---
        net_people_total = 0 # Initialize for Eagle-Eye mode
        if self.mode == 'Eagle‑Eye':
            net_people_total = len(current_persons_detections)
            # Draw all detections for Eagle-Eye mode
            for x1, y1, x2, y2, label in detections:
                color = (0, 255, 0) if label == 'person' else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
        elif self.mode == 'Lane Counter':
            for obj_id, obj_data in self.tracked_objects.items():
                x1, y1, x2, y2 = obj_data['bbox']
                cx, cy = obj_data['centroid']

                # Determine lane for drawing and counting
                if cx < self.lane_split_x:
                    obj_data['lane'] = 'enter'
                    color = (255, 255, 0) # Cyan for enter lane
                else:
                    obj_data['lane'] = 'exit'
                    color = (0, 165, 255) # Orange for exit lane

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
                cv2.putText(frame, obj_data['lane'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)

                # Entry Logic: Person in enter lane, bottom crosses entry threshold, not yet counted in
                if obj_data['lane'] == 'enter' and y2 > self.entry_y_threshold and not obj_data['counted_in']:
                    self.in_count += 1
                    obj_data['counted_in'] = True
                    # Optional: Mark as 'finished' or remove if no longer needed to track after counting
                    # For simplicity, we keep tracking but prevent re-counting

                # Exit Logic: Person in exit lane, top crosses exit threshold, not yet counted out
                if obj_data['lane'] == 'exit' and y1 < self.exit_y_threshold and not obj_data['counted_out']:
                    self.out_count += 1
                    obj_data['counted_out'] = True
                    # Optional: Mark as 'finished' or remove if no longer needed to track after counting

            net_people_total = self.in_count - self.out_count

        return self.in_count, self.out_count, net_people_total, frame

