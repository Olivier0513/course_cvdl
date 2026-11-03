# hw1_1
## 1.1 Corner Detection 
### Find Chessboard Corners
使用 `cv2.findChessboardCorners()` 偵測棋盤格的內部角點。
#### Function
```python
retval, corners = cv2.findChessboardCorners(
    image,
    patternSize,
    flags=None
)
```
#### Parameters
| Parameter | Description |
|---|---|
| `image` | 輸入的灰階影像（Grayscale Image） |
| `patternSize` | 棋盤格內部角點的數量 `(columns, rows)` |
| `flags` | 可選的偵測參數 |
#### Return Values
| Return | Description |
|---|---|
| `ret` | 是否成功偵測到所有角點，成功為 `True` |
| `corners` | 偵測到的角點座標，形狀通常為 `(N, 1, 2)` |
### Draw Chessboard Corners
使用 `cv2.drawChessboardCorners()` 畫出棋盤格的內部角點。
```python
retval = cv2.drawChessboardCorners(
    image,
    patternSize,
    corners,
    patternWasFound
)
```
## 1.2 Find the Intrinsic Matrix
使用 `cv2.calibrateCamera()` 計算相機的內參矩陣。
#### Function
```python
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints,
    imagePoints,
    imageSize,
    cameraMatrix=None,
    distCoeffs=None,
    flags=None,
    criteria=None
)
```
#### Parameters
| Parameter | Description |
|---|---|
| `objectPoints` | 每張影像對應的 3D 世界座標，通常使用棋盤格角點建立 |
| `imagePoints` | 每張影像偵測到的 2D 影像座標，通常由 `findChessboardCorners()` 取得 |
| `imageSize` | 影像尺寸，格式為 `(width, height)` |
| `cameraMatrix` | 初始相機內參矩陣，若未提供則由函式估算 |
| `distCoeffs` | 初始鏡頭畸變係數 |
| `flags` | 相機校正的設定選項 |
| `criteria` | 迭代終止條件，包含最大迭代次數與精度 |
#### Return Values
| Return | Description |
|---|---|
| `retval` | 重投影誤差（Re-projection Error），用來評估校正結果 |
| `cameraMatrix` | 相機內參矩陣（Intrinsic Matrix） |
| `distCoeffs` | 鏡頭畸變係數（Distortion Coefficients） |
| `rvecs` | 每張影像對應的旋轉向量（Rotation Vectors） |
| `tvecs` | 每張影像對應的平移向量（Translation Vectors） |
## 1.3 Find the Extrinsic Matrix 
使用 `cv2.calibrateCamera()` 透過`rvec`和`tvec`組成每張圖片的外參矩陣。
## 1.4 Find the Distortion Matrix
使用 `cv2.calibrateCamera()` 透過`distCoeffs`得到鏡頭畸變係數。
## 1.5 Show the undistorted result
使用 `cv2.undistort()` 移除鏡頭造成的徑向畸變（Radial Distortion）與切向畸變（Tangential Distortion），將原始影像轉換為校正後的影像。
#### Function
```python
dst = cv2.undistort(
    src,
    cameraMatrix,
    distCoeffs,
    dst=None,
    newCameraMatrix=None
)
```
#### Parameters
| Parameter | Description |
|---|---|
| `src` | 輸入的原始影像（Original Image） |
| `cameraMatrix` | 相機內參矩陣（Intrinsic Matrix），由 `calibrateCamera()` 取得 |
| `distCoeffs` | 鏡頭畸變係數（Distortion Coefficients），由 `calibrateCamera()` 取得 |
| `dst` | 輸出影像，通常不需要自行指定 |
| `newCameraMatrix` | 新的相機內參矩陣，用來控制校正後影像的視野與裁切範圍 |
#### Return Value
| Return | Description |
|---|---|
| `dst` | 去除鏡頭畸變後的輸出影像 |
# hw1_2
## 2.1 Show Letter
### Letter Library
```
A: !!opencv-matrix
   rows: 5
   cols: 2
   dt: "3i"
   data: [  2, 0, 0, 2, 1, 0, 
            2, 1, 0, 1, 2, 0, 
            1, 2, 0, 0, 1, 0, 
            0, 1, 0, 0, 0, 0, 
            2, 1, 0, 0, 1, 0 ]
```
`data` 記錄組成字母 A 的線段座標，共 5 組。
### Calculate shift amount
計算輸入的每個字母要放在棋盤的哪裡，每個位置偏移量不同。
```python
if j == 0:
    dy = 5
    dx = 7
elif j == 1:
    dy = 5
    dx = 4           
elif j == 2:
    dy = 5
    dx = 1
elif j == 3:
    dy = 2
    dx = 7
elif j == 4:
    dy = 2
    dx = 4
elif j == 5:
    dy = 2
    dx = 1
```
### Shift each letter
將每個字母的每個線段的兩個端點都移到對應位置。
```python
ch[k][0][0] = ch[k][0][0] + dx
ch[k][0][1] = ch[k][0][1] + dy
ch[k][1][0] = ch[k][1][0] + dx
ch[k][1][1] = ch[k][1][1] + dy
```
### Project 3D Points to Image Plane
使用 `cv2.projectPoints()` 將 3D 物件座標透過相機的內參、外參及畸變係數，投影至 2D 影像座標。
#### Function
```python
imagePoints, jacobian = cv2.projectPoints(
    objectPoints,
    rvec,
    tvec,
    cameraMatrix,
    distCoeffs
)
```
#### Parameters
| Parameter | Description |
|---|---|
| `objectPoints` | 輸入的 3D 世界座標點，通常使用 `np.float32` 或 `np.float64` |
| `rvec` | 旋轉向量（Rotation Vector），描述物件座標系與相機座標系之間的旋轉 |
| `tvec` | 平移向量（Translation Vector），描述物件座標系與相機座標系之間的平移 |
| `cameraMatrix` | 相機內參矩陣（Intrinsic Matrix） |
| `distCoeffs` | 鏡頭畸變係數（Distortion Coefficients） |
#### Return Values
| Return | Description |
|---|---|
| `imagePoints` | 投影後的 2D 影像座標，通常形狀為 `(N, 1, 2)` |
| `jacobian` | 投影座標對輸入參數的 Jacobian Matrix（雅可比矩陣），用於計算投影結果對參數變化的敏感度 |
### Draw Words on Chessboard
根據計算出的 `imagePoints` 畫出字母。
```python
imgs[i] = cv2.line(imgs[i],(int(imgpoints[0][0][0]),int(imgpoints[0][0][1])),(int(imgpoints[1][0][0]),int(imgpoints[1][0][1])), (0, 0, 255), 5)
```
## 2.2 Show Letter Vertical
與題目 2.1 操作相同，只是使用不同的 library。
# hw1_3
## 3.1 Stereo Disparity Map
使用 `cv2.StereoBM_create()` 透過 `numDisparities` 和 `blockSize` 求出一個 StereoBM 演算法物件，再使用 `stereoBM.compute()` 計算左右影像的視差圖。
#### Function
```python
stereo = cv2.StereoBM_create(
    numDisparities=0,
    blockSize=21
)
```
#### Parameters
| Parameter | Description |
|---|---|
| `numDisparities` | 搜尋的視差範圍，必須是 16 的倍數，且大於 0 |
| `blockSize` | 區塊匹配的視窗大小，必須是大於等於 5 的奇數，通常介於 5 到 255 |
## 3.2 Checking the Disparity Value
在左邊點選任意位置，若其 `disparity` 為正，則在右圖顯示對應位置。