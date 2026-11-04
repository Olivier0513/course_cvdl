# hw1_4
## 4.1 Keypoints
使用 `sift.detect()` 來找出圖片中的keypoints。
```python
sift = cv2.SIFT_create()
keypoints = sift.detect(self.img_gray_1, None)
```
使用 `cv2.drawKeypoints()` 畫出圖片中的keypoints。
```python
result = cv2.drawKeypoints(self.img_gray_1, keypoints, 0, (0,255,0), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

```
## 4.2 Matched Keypoints
### Detect keypoints and compute descriptors
使用 `cv2.detectAndCompute()` 分別偵測兩張影像中的特徵點（Keypoints），並計算對應的特徵描述子（Descriptors）。
```python
key1, des1 = sift.detectAndCompute(self.img_gray_1, None)
key2, des2 = sift.detectAndCompute(self.img_gray_2, None)
```
### Feature Matching
使用 `cv2.BFMatcher()` 建立暴力特徵匹配器（Brute-Force Matcher），比較兩張影像的特徵描述子（Descriptors），透過 KNN 找出每個特徵點最相似的兩個匹配結果。
```python
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
```
### Lowe's Ratio Test
使用 `Lowe's Ratio Test` 比較最近匹配（Best Match）與第二近匹配（Second Best Match）的距離，篩選較可靠的特徵匹配結果。
```python
if m.distance < 0.5 * n.distance:
    good_matches[i] = [1, 0]
```