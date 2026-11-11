# h2_1 Background Subtraction
### Build Background Model
Use first 25 frames to build the background model
### Mean and Standard Deviation
Each pixel will have a mean and a standard deviation
```python
mean_array = first_25_array.mean(axis=0)
std_array = first_25_array.std(axis=0)
# if standard deviation is less then 5, set to 5
std_array[np.where(std_array<5)] = 5
```
### Mask
- If Frame(x , y) - mean > std*5 ,then mask(x , y) = 255
- If Frame(x , y) - mean <= std*5 ,then mask(x , y) = 0
# h2_2 Optical Flow
## 2.1 Preprocessing
### Detect Blobs
use `cv2.SimpleBlobDetector_create()` to detect blobs
```python
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(frame)
```
### Mark Circles
Mark circles with a red square bounding box and cross mark
```python
frame_keypoints = cv2.line(frame,(x,y-6),(x,y+6),(0,0,255))
frame_keypoints = cv2.line(frame,(x-6,y),(x+6,y),(0,0,255))
frame_keypoints = cv2.rectangle(frame,(x-6,y-6),(x+6,y+6),(0,0,255))
```
## 2.2 Video Tracking
Calculate Optical Flow
```python
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
```
# h2_3 Perspective Transform
### Detect ArUco markers
Detect four ArUco markers position and markerID, each marker has 4 corner positions and 1 markerID
```python
arucoDictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
arucoParameters = cv2.aruco.DetectorParameters_create()
corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, arucoDictionary, parameters = arucoParameters)
```
### Perspective Transform
Calculate Homography Matrix and do perspective transform
```python
homo, status = cv2.findHomography(np.float32(projectList), np.float32(pts_dst))
im_out = cv2.warpPerspective(projectImg, homo, (frame.shape[1],frame.shape[0]))
```
# h2_4 PCA
## 4.1 Image Reconstruction
Reduce the component from 30 to 27
```python
pca = PCA(n_components = 27)
```
Analyses 2D data using PCA
```python
reducedImg = pca.fit_transform(data)
```
Reconstruct image from reduced data
```python
self.reconstructedImg = pca.inverse_transform(reducedImg)
```
## 4.2 Compute the reconstruction error
Computing the reconstruction error
```python
err = np.sqrt(np.sum((ori_gray-rc_gray)**2))
```