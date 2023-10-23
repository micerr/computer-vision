# Theoretical introduction

Camera calibration is based on the pinhole camera model that approximate how lights travels between the scene and the camera sensor. There are two sets of parameters:

- The **extrinsic parameters** define the camera position and orientation relative to the world coordinates system.
- The **Intrinsic parameters** define the camera sensor and lens parameters.

There are 3 different coordinates systems:

- The world 3D coordinates system
- The camera 3D coordinates system
- The 2D pixel positions where 3D positions are being projected.


## Extrinsic parameters
The extrinsic parameters defines the transformations from the world 3D coordinates $$\left[x_O,y_O,z_O\right]^T$$ to the camera 3D coordinates $$\left[x_C,y_C,z_C\right]^T$$. The camera 3D coordinates system has the following **conventions**:

- The point $(0,0,0)$ is the center of projection of the camera and is called the *principal point*.
- The $z$ axis of the camera points *towards the scene*, the $x$ axis is along the sensor width pointing towards the right, and the $y$ axis is along the sensor height pointing towards the bottom.

The camera coordinates system is therefore a *transformation* of the world coordinates systems with:

- A **rotation** defined by a rotation matrix $R$ using euler angles in a right-hand orthogonal system. The rotation is applied to the world coordinates system to obtain the camera orientation.
- A **translation** defined by a translation vector $T$ representing the position of the center of the world in the camera coordinates system !

Hence,

$$\lambda\left[\begin{matrix}x_C\\ y_C\\ z_C\\ 1\end{matrix}\right] = \left[\begin{matrix}
R_{3\times 3} & T_{3\times 1}\\{\bf 0}_{1\times 3}&1
\end{matrix}\right] \left[\begin{matrix}x_O\\y_O\\z_O\\1\end{matrix}\right]$$

**Important notes:**

- The rotation matrix represents a passive (or alias) transformation because it's the coordinates system that rotates and not the objects.
- Euler angles define a 3D rotation starting with a rotation around $x$ followed by a rotation around $y$ followed by a rotation around $z$ (the order matters).
- If $T$ is expressed in the camera coordinate system, the position of the camera expressed in world coordinates system is $C:=-R^{-1}T = -R^{T}T$ (since $R$ is a rotation matrix).


## Intrinsic parameters

The intrinsic parameters defines the transformation between the 3D coordinates relative to the camera center $\left[x_C,y_C,z_C\right]^T$ and the 2D coordinates in the camera sensor $\left[i,j\right]^T$. This transformation is called a *projection* and includes:

- the scale produced by the focal length, with $f$ being the distance between the camera center and the plane on which the image is projected.
- the scale factors $(m_x,m_y)$ relating pixels units to distance units (usually $m_x=m_y$ because pixels are squares).
- the translation from the camera _principal point_ to a top-left origin, with $(u_0,v_0)$ being the position of the *principal point* expressed in the image coordinates system.
- a skew coefficient $\gamma$ between the $x$ and $y$ axis in the sensor (usually $\gamma=0$ because pixels are squares).

Those transformations can be aggregated in one single matrix called **camera matrix**:

$$K := \left[\begin{matrix}f\cdot m_x & \gamma & u_0 \\ 0 & f\cdot m_y & v_0 \\ 0 & 0 & 1\end{matrix}\right]$$

Therefore,
$$\lambda\left[\begin{matrix}i\\ j\\ 1\end{matrix}\right]= \left[\begin{matrix}K_{3\times 3}&{\bf 0}_{3\times 1}\end{matrix}\right]\left[\begin{matrix}x_C\\y_C\\z_C\\1\end{matrix}\right]$$

**Notes:**

- The width and height of the image are to be added to those parameters and delimits the sensor width and height in pixels.
- When applying the **direct** projection of a given 3D point, different values of $\lambda$ will always give the **same** 2D point.
- When applying the **inverse** projection on a given 2D point, different values of $\lambda$ will give **different** 3D points.

This is obvious when simplifying the relation between the two points (The column ${\bf 0}_{3\times 1}$ cancels the homogenous component of the 3D point):

$$\lambda\left[\begin{matrix}i\\j\\1\end{matrix}\right]= \left[\begin{matrix}K_{3\times 3}\end{matrix}\right]\left[\begin{matrix}x_C\\y_C\\z_C\end{matrix}\right]$$

The 2D vector in homogenous coordinates is not affected by the value of $\lambda$, while the 3D vector is.



## Projection model

Therefore, by combining

- the transformation from the world coordinates system to the camera coordinates system (defined by $R$ and $T$)
- with the projection from the camera coordinates system to the image pixels (defined by $K$),

We have a projection model allowing to compute the coordinates of a 2D point in the image $\left(i,j\right)$ from a 3D point in the real world $\left(x,y,z\right)$ described by the matrix $P$:
$$P := \left[\begin{matrix}K_{3\times 3}&{\bf 0}_{3\times 1}\end{matrix}\right] \left[\begin{matrix}R_{3\times 3}&T_{3\times 1}\\{\bf 0}_{1\times 3}&1\end{matrix}\right]=K_{3\times 3}\left[\begin{matrix}R_{3\times 3}&T_{3\times 1}\end{matrix}\right]$$

The opposite operation requires to invert $P$ and is done by pseudo-inverse inversion because $P$ is rectangular.


## Limitations
We need a model of the distortions brought by the lens:

- **radial** distortion cause lines far from principal point to look distorted.
- **tangential** distortion: occur when lens is not prefectly align with the $z$ axis of the camera.

Many lens distortion models exist. Here, we will use the model used in [opencv](https://docs.opencv.org/3.1.0/d4/d94/tutorial_camera_calibration.html).


## Distortion models

There are actually 2 different models:

- The direct model that applies distortion: find the 2D image (distorted) coordinates of a 3D point given its 2D projected (undistorted) coordinates.
- The inverse model that rectifies distortion: find the 2D (undistorted) coordinate allowing to project a 2D (distorted) image coordinate into 3D.

.. image:: https://github.com/ispgroupucl/calib3d/blob/main/assets/distortion_steps.png?raw=true

### Direct model: "distort"

The following relationships allows to compute the distorted 2D coordinates $(x_d,y_d)$ on an image from the 2D coordinates provided by the linear projection model $(x_u,y_u)$. Those coordinates are expressed in the camera coordinate system, i.e. they are distances (not pixels) and the point $(0,0)$ is the principal point of the camera.

$$\begin{align}
    x_d &= x_u\overbrace{\left(1 + k_1 {r_u}^2 + k_2 {r_u}^4 + k_3 {r_u}^6+\cdots\right)}^{\text{radial component}} + \overbrace{\left[2p_1 x_uy_u + p_2\left({r_u}^2 + 2{x_u}^2\right)\right]}^{\text{tangeantial component}}\\
    y_d &= y_u\left(1 + k_1 {r_u}^2 + k_2 {r_u}^4 + k_3 {r_u}^6+\cdots\right) + \left[2p_2 x_uy_u + p_1\left({r_u}^2 + 2{y_u}^2\right)\right]
\end{align}$$

Where:

- $k_1, k_2, k_3, \cdots$  are the radial distortion coefficients
- $t_1, t_2$ are the tangential distortion coefficients
- ${r_u}^2 := {x_u}^2 + {y_u}^2$

We usually use only 3 radial distortion coefficients, which makes a total of 5 coefficients. Those coefficients are found by running an optimisation algorithm on a set of 2D point - 3D point relations as we did with `cv2.calibrateCamera`. They are stored in the `kc` vector.

### Inverse model: "rectify"

The distortion operation cannot be inverted analitically using the coefficients $k_1$, $k_2$, $k_3$, $p_1$, $p_2$ (i.e. have $x_u=f_{k_{1,2,3},p_{1,2}}(x_d,y_d)$ and $y_u=f_{k_{1,2,3},p_{1,2}}(x_d,y_d)$). We either need another model with another set of coefficients, or make an approximation.

Here, we will use the following approximation: We will assume that the distortion at point $(x_d,y_d)$ would be the same that distortion at point $(x_u,y_u)$ ! Therefore:

$$\left\{\begin{align}
    2p_1 x_uy_u + p_2\left({r_u}^2 + 2{x_u}^2\right) &\approx 2p_1 x_dy_d + p_2\left({r_d}^2 + 2{x_d}^2\right)\\
    2p_2 x_uy_u + p_1\left({r_u}^2 + 2{y_u}^2\right) &\approx 2p_2 x_dy_d + p_1\left({r_d}^2 + 2{y_d}^2\right) \\
    \left(1 + k_1 {r_u}^2 + k_2 {r_u}^4 + k_3 {r_u}^6+\cdots\right) &\approx \left(1 + k_1 {r_d}^2 + k_2 {r_d}^4 + k_3 {r_d}^6+\cdots\right)
   \end{align}\right.
    $$

If this approximation holds, it's much easier to get an analytical expression of $x_u=f_{k_{1,2,3},p_{1,2}}(x_d,y_d)$ and $y_u=f_{k_{1,2,3},p_{1,2}}(x_d,y_d)$.


# Implementation

This library defines a [Calib](./calib3d.calib#Calib) object to represent a calibrated camera. Its constructor receives in arguments the intrinsic and extrinsic parameters:

- image dimensions `width` and `height`,
- the translation vector `T`,
- the rotation matrix `R`,
- the camera matrix `K`.

The method `project_3D_to_2D` allows to compute the position in the image of a 3D point in the world. The opposite operation `project_2D_to_3D` requires an additional parameter `Z` that tells the $z$ coordinate of the 3D point.