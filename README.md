# 🤖 6-axis-articulated-Robot-Simulator
Developed using Python, PyQt5, and Matplotlib, this **6-axis-articulated-Robot-Simulator** is a powerful engineering tool that provides real-time calculation and 3D visualization of Forward Kinematics (FK) and Inverse Kinematics (IK) through an intuitive GUI based on the Damped Least Squares (DLS) method. It also features point-teaching capabilities and an advanced cylinder-based obstacle avoidance trajectory generation algorithm.

---

## 📺 Project Walkthrough

Watch the simulator in action and see the 'Vibe Coding' development process in the YouTube video below:

[![6-DOF Robot Simulator Demo](https://img.youtube.com/vi/WON2r15oeLg/0.jpg)](https://www.youtube.com/watch?v=WON2r15oeLg)
> **Click the image above to watch the "Vibe Coding" process for this 6-axis industrial robot.**

---

## ✨ Key Features

* **Real-time Kinematics & 3D Visualization:**
  * Dynamic rendering of the 6-axis arm configuration based on Denavit-Hartenberg (DH) parameters.
  * Instant numerical solving of Forward Kinematics (Angles) and DLS-based Inverse Kinematics (Coordinates + RPY Orientation).
* **Interactive Control Panels:**
  * **Jog Buttons:** Multi-axis step-by-step jog controls for individual joints (J1 - J6) and Cartesian space ($X, Y, Z, \text{Roll}, \text{Pitch}, \text{Yaw}$).
  * **Precision Sliders & Direct Inputs:** Instantly manipulate target states with responsive sliders and direct textbox updates.
* **Teaching & Playback:**
  * Save spatial target poses (Point A, Point B) capturing both position and orientation matrices.
  * Execute continuous reciprocating path simulations between taught points.
* **Obstacle Avoidance Trajectory Generation:**
  * Define virtual cylindrical obstacles inside the 3D workspace using coordinates, physical dimensions, and safety margins.
  * Automatically compute a collision-free path that maintains optimal manipulability while completely bypassing the defined clearance zones.

---

## 🚀 How to Run

1. **Ensure Python is installed.**
2. **Install the required GUI and visualization libraries:**
   ```bash
   pip install numpy matplotlib PyQt5

```

3. **Run the script:**
```bash
python 6_axis_articulated_Robot_Simulator.py
```



---

### 🌐 Multi-Language Support (20 Languages)

To support engineers, educators, and creators worldwide, this content provides both **Voiceovers and Subtitles in 20 different languages**.

You can easily adjust the audio track and subtitle settings in the YouTube player to watch in your preferred language:

* **Dutch** (Nederlands)
* **German** (Deutsch)
* **Russian** (Русский)
* **Vietnamese** (Tiếng Việt)
* **Bengali** (বাংলা)
* **Spanish** (Español)
* **Arabic** (العربية)
* **English**
* **Ukrainian** (Українська)
* **Italian** (Italiano)
* **Indonesian** (Bahasa Indonesia)
* **Japanese** (日本語)
* **Chinese (Traditional)** (繁體中文)
* **Thai** (ไทย)
* **Turkish** (Türkçe)
* **Portuguese** (Português)
* **Polish** (Polski)
* **French** (Français)
* **Hindi** (हिन्दी)

**Author:** **SUCHEOL LEE** (Lee Sucheol Robotics Lab.)

**Expertise:** Robotic Mechanism Design, Kinematics, CAD 19+ Years

```

```
