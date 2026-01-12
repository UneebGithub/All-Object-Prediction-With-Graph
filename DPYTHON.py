import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Dashboard setup
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'facecolor': 'black'})
ax1 = fig.add_subplot(131, projection='3d')


model = YOLO('yolov8n-seg.pt') 


target_classes = [0, 2]


my_car_ref_area_center_x = 320 
my_car_ref_area_center_y = 600 
my_car_search_radius = 200 

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    
    results = model.track(frame, persist=True, classes=target_classes)

    ax1.cla(); ax2.cla(); ax3.cla()

    
    cv2.rectangle(frame, (my_car_ref_area_center_x - 30, my_car_ref_area_center_y - 50),
                  (my_car_ref_area_center_x + 30, my_car_ref_area_center_y), (0, 255, 0), -1)
    cv2.putText(frame, "MY CAR", (my_car_ref_area_center_x - 25, my_car_ref_area_center_y - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    
    relevant_people = []

    if results[0].boxes.id is not None:
        seg_img = results[0].plot(labels=True, boxes=False)
        
        
        cv2.circle(seg_img, (my_car_ref_area_center_x, my_car_ref_area_center_y), 
                   my_car_search_radius, (255, 255, 0), 2)
        cv2.putText(seg_img, "People Search Area", 
                    (my_car_ref_area_center_x + my_car_search_radius // 2, my_car_ref_area_center_y - my_car_search_radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        ax2.imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
        ax2.set_title("Live Detection & Focus Area", color='white')

        for box in results[0].boxes:
            x_center, y_center, w, h = box.xywh[0].cpu().numpy()
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            
            if cls == 0: 

                distance = np.sqrt((x_center - my_car_ref_area_center_x)**2 + 
                                   (y_center - my_car_ref_area_center_y)**2)
                
                if distance < my_car_search_radius:
                    relevant_people.append((x_center, y_center, w, h, track_id))
                    
                    cv2.rectangle(frame, (int(x_center-w/2), int(y_center-h/2)),
                                  (int(x_center+w/2), int(y_center+h/2)), (0, 255, 255), 3)
                    cv2.putText(frame, f"Person {track_id} - NEAR CAR", 
                                (int(x_center-w/2), int(y_center-h/2)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            
            color = 'deeppink' if cls == 0 else 'cyan' 
            ax3.scatter(x_center, y_center, c=color, s=50)
            

            ax3.arrow(x_center, y_center, 30, -20, head_width=10, color='red')

    ax3.set_xlim(0, frame.shape[1]); ax3.set_ylim(frame.shape[0], 0)
    ax3.set_title("Top-View: All Object Prediction", color='white')
    ax3.grid(True, color='gray', linestyle='--', alpha=0.5)

    if relevant_people:
        points_x = [p[0] for p in relevant_people]
        points_y = [p[1] for p in relevant_people]
        
        points_z = [(frame.shape[0] - p[1]) / 100 for p in relevant_people] 
        
        ax1.scatter(points_x, points_y, points_z, c='yellow', s=50, edgecolors='white')
        ax1.set_title("3D View: People Near My Car", color='white')
        ax1.set_xlabel("X (Width)", color='white')
        ax1.set_ylabel("Y (Depth)", color='white')
        ax1.set_zlabel("Z (Height)", color='white')
        ax1.tick_params(colors='white') 
        ax1.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)) 
        ax1.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax1.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    else:
        ax1.set_title("3D View: No People Near Car", color='gray')


    plt.pause(0.01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.close()