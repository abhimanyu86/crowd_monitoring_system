import time
import smtplib
import ssl
from email.message import EmailMessage
import cv2  # OpenCV needed for frame annotation
from settings import EMAIL

class AlertManager:
    def __init__(self, capacity, cooldown=10):
        """
        Initialize AlertManager with capacity limit and cooldown period (in seconds) 
        to avoid sending repeated emails too frequently.
        """
        self.capacity = capacity
        self.cooldown = cooldown
        self.last_sent = 0
        self.alert_active = False  # For capacity alerts
        self.restricted_alert_active = False # For restricted item alerts

    def send_email(self, subject, body):
        current_time = time.time()
        if current_time - self.last_sent < self.cooldown:
            print("Cooldown active: skipping email send.")
            return False  # Skip if cooldown not passed
        
        # Ensure body is not None or empty, though current logic should prevent this.
        final_body = body
        if not final_body or final_body.strip() == "":
            print(f"Warning: Email body was empty or whitespace for subject '{subject}'. Using a default body.")
            final_body = f"Alert triggered for: {subject}. (Original body was unexpectedly empty)"

        self.last_sent = current_time

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL['sender']
        msg['To'] = EMAIL['recipient']
        msg.set_content(final_body)

        context = ssl.create_default_context()

        try:
            with smtplib.SMTP_SSL(EMAIL['smtp_server'], EMAIL['smtp_port'], context=context) as server:
                server.login(EMAIL['sender'], EMAIL['password'])
                server.send_message(msg)
            print("Alert email sent successfully.")
            return True
        except Exception as e:
            print(f"Failed to send alert email: {e}")
            return False

    def check_capacity(self, current_count):
        if current_count > self.capacity:
            if not self.alert_active:
                subject = "Capacity Alert"
                details = f"Current count of {current_count} has exceeded the set capacity of {self.capacity}."
                body = f"Alert Type: Capacity Exceeded\n\nDetails: {details}"
                if self.send_email(subject, body):
                    self.alert_active = True
            return True
        else:
            # Reset alert status if capacity no longer exceeded
            self.alert_active = False
            return False

    def handle_capacity(self, frame, current_count):
        """
        Check capacity and send alert if needed.
        Annotate the frame with warning text if capacity exceeded.
        Returns True if alert active, else False.
        """
        alert_triggered = self.check_capacity(current_count)
        if alert_triggered:
            # Put red alert text on frame at top-left corner
            cv2.putText(frame, "CAPACITY EXCEEDED!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        return alert_triggered

    def handle_restricted(self, frame, detections, restricted_items_lower):
        """
        Check for restricted items and send alert if needed.
        Annotate the frame with warning text if restricted items are detected.
        `restricted_items_lower` should be a list of lowercase item names.
        Returns True if a restricted item is detected, else False.
        """
        found_restricted_labels_in_frame = []
        is_restricted_item_present_this_frame = False

        for *_, label_detected_original in detections: # Detections are (x1,y1,x2,y2, label)
            label_detected_lower = label_detected_original.lower()
            if label_detected_lower in restricted_items_lower:
                is_restricted_item_present_this_frame = True
                if label_detected_original not in found_restricted_labels_in_frame:
                    found_restricted_labels_in_frame.append(label_detected_original)
        
        if is_restricted_item_present_this_frame:
            if not self.restricted_alert_active:
                subject = "Restricted Item Alert"
                
                # Construct the detailed part of the body
                if found_restricted_labels_in_frame:
                    details = f"The following restricted item(s) were detected: {', '.join(found_restricted_labels_in_frame)}."
                else:
                    # This case implies a restricted item was flagged but no specific labels were collected.
                    # This safeguard ensures the email still provides context.
                    details = "One or more restricted items were detected, but their specific names could not be listed from the current frame's detections."
                    print("Warning: Restricted item alert triggered, but found_restricted_labels_in_frame was empty.")
                
                body = f"Alert Type: Restricted Item\n\nDetails: {details}"
                
                if self.send_email(subject, body):
                    self.restricted_alert_active = True
            
            # Annotate frame if a restricted item is present in this frame
            cv2.putText(frame, "RESTRICTED ITEM DETECTED!", (50, 100), # Positioned below capacity alert
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA) # Orange color
            return True # Indicates a restricted item condition is active (based on current frame)
        else:
            # Reset restricted alert email status if no restricted items are currently detected
            self.restricted_alert_active = False
            return False