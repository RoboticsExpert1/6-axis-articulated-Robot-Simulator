import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSlider, 
                             QLineEdit, QGridLayout, QMessageBox, QSplitter, QHeaderView, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class RobotArm:
    def __init__(self):
        self.dh_params = np.array([
            [0.0, 300.0, 0.0, -90.0],
            [0.0, 0.0, 250.0, 0.0],
            [0.0, 0.0, 50.0, -90.0],
            [0.0, 200.0, 0.0, 90.0],
            [0.0, 0.0, 0.0, -90.0],
            [0.0, 100.0, 0.0, 0.0]
        ])
        self.num_joints = 6
        self.q = np.zeros(6)

    def dh_transform(self, theta, d, a, alpha):
        theta = np.radians(theta)
        alpha = np.radians(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])

    def forward_kinematics(self, q=None):
        if q is None:
            q = self.q
        
        T = np.eye(4)
        positions = [T[:3, 3]]
        z_axes = [T[:3, 2]]
        
        for i in range(self.num_joints):
            theta_offset, d, a, alpha = self.dh_params[i]
            theta = np.degrees(q[i]) + theta_offset
            T_i = self.dh_transform(theta, d, a, alpha)
            T = T @ T_i
            positions.append(T[:3, 3])
            z_axes.append(T[:3, 2])
            
        return T, np.array(positions), np.array(z_axes)

    def get_jacobian(self, q):
        T_end, positions, z_axes = self.forward_kinematics(q)
        pos_end = positions[-1]
        
        J = np.zeros((6, self.num_joints))
        for i in range(self.num_joints):
            z_i = z_axes[i]
            p_i = positions[i]
            J[:3, i] = np.cross(z_i, pos_end - p_i)
            J[3:, i] = z_i
        return J
        
    def get_manipulability(self, q):
        J = self.get_jacobian(q)
        return np.sqrt(max(0, np.linalg.det(J @ J.T)))

    def inverse_kinematics_dls(self, target_pos, target_rot, current_q, max_iter=200, tol=1e-2):
        q = np.copy(current_q)
        damping_factor = 0.1 
        
        for _ in range(max_iter):
            T_current, _, _ = self.forward_kinematics(q)
            current_pos = T_current[:3, 3]
            current_rot = T_current[:3, :3]
            
            pos_err = target_pos - current_pos
            rot_err = 0.5 * (np.cross(current_rot[:, 0], target_rot[:, 0]) +
                             np.cross(current_rot[:, 1], target_rot[:, 1]) +
                             np.cross(current_rot[:, 2], target_rot[:, 2]))
            
            err = np.concatenate((pos_err, rot_err))
            
            if np.linalg.norm(err) < tol:
                return q, True
                
            J = self.get_jacobian(q)
            J_T = J.T
            
            # Less excessive singularity penalty, prevents getting stuck
            manipulability = np.sqrt(max(0, np.linalg.det(J @ J_T)))
            lambda_sq = damping_factor ** 2
            if manipulability < 0.01:
                lambda_sq += 0.01
                
            J_dls = J_T @ np.linalg.inv(J @ J_T + lambda_sq * np.eye(6))
            dq = J_dls @ err
            
            # Clip joint step size to prevent explosions near singularities
            max_step = 0.1 
            step_norm = np.linalg.norm(dq)
            if step_norm > max_step:
                dq = dq * (max_step / step_norm)
                
            q += dq
            
        # Soft success: If exact tolerance wasn't met but it's very close (e.g. within 5mm / 0.1rad)
        if np.linalg.norm(err[:3]) < 5.0 and np.linalg.norm(err[3:]) < 0.1:
            return q, True
            
        return q, False

def euler_to_matrix(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def matrix_to_euler(R):
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return x, y, z

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6-DOF Robot Kinematics GUI (Obstacle Avoidance & Improved IK)")
        self.resize(1500, 950)
        
        self.robot = RobotArm()
        self.fk_controls = []
        self.ik_controls = []
        
        self.point_A = None
        self.point_B = None
        self.trajectory_points = None
        self.drawn_trajectory = None
        
        self.is_reciprocating = False
        self.current_target = 'B'
        
        self.obstacle_active = False
        
        self.init_ui()
        self.update_fk_from_robot()
        self.update_plot()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- LEFT PANEL (Input) wrapped in QScrollArea ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        
        # DH Parameters
        left_layout.addWidget(QLabel("<b>DH Parameters (theta_off, d, a, alpha)</b>"))
        self.dh_table = QTableWidget(6, 4)
        self.dh_table.setHorizontalHeaderLabels(["Theta Offset", "d", "a", "Alpha"])
        self.dh_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dh_table.setMaximumHeight(200)
        for i in range(6):
            for j in range(4):
                self.dh_table.setItem(i, j, QTableWidgetItem(str(self.robot.dh_params[i][j])))
        left_layout.addWidget(self.dh_table)
        
        btn_apply_dh = QPushButton("Apply DH Parameters")
        btn_apply_dh.clicked.connect(self.apply_dh_params)
        left_layout.addWidget(btn_apply_dh)
        
        # Jog Step
        jog_layout = QHBoxLayout()
        jog_layout.addWidget(QLabel("<b>Jog Step:</b>"))
        self.jog_step_input = QLineEdit("5.0")
        jog_layout.addWidget(self.jog_step_input)
        left_layout.addLayout(jog_layout)
        
        # FK Controls
        left_layout.addWidget(QLabel("<b>Forward Kinematics (J1 - J6) [deg]</b>"))
        for i in range(6):
            ctrl = self.create_slider_row(f"J{i+1}", -180, 180, self.fk_changed, self.jog_fk)
            self.fk_controls.append(ctrl)
            left_layout.addLayout(ctrl['layout'])
            
        # IK Controls
        left_layout.addWidget(QLabel("<b>Inverse Kinematics (X,Y,Z, R,P,Y)</b>"))
        ik_labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        ik_ranges = [(-1000, 1000), (-1000, 1000), (-1000, 1000), (-180, 180), (-180, 180), (-180, 180)]
        for i in range(6):
            ctrl = self.create_slider_row(ik_labels[i], ik_ranges[i][0], ik_ranges[i][1], self.ik_changed, self.jog_ik)
            self.ik_controls.append(ctrl)
            left_layout.addLayout(ctrl['layout'])

        # Trajectory
        left_layout.addWidget(QLabel("<b>Trajectory Planning</b>"))
        grid_traj = QGridLayout()
        btn_save_A = QPushButton("Save Point A")
        btn_save_A.clicked.connect(lambda: self.save_point('A'))
        self.btn_move_A = QPushButton("Move to Point A")
        self.btn_move_A.clicked.connect(lambda: self.move_to_point('A'))
        
        btn_save_B = QPushButton("Save Point B")
        btn_save_B.clicked.connect(lambda: self.save_point('B'))
        self.btn_move_B = QPushButton("Move to Point B")
        self.btn_move_B.clicked.connect(lambda: self.move_to_point('B'))
        
        self.btn_reciprocate = QPushButton("Reciprocating Motion")
        self.btn_reciprocate.clicked.connect(self.toggle_reciprocating)
        self.btn_reciprocate.setEnabled(False)
        
        grid_traj.addWidget(btn_save_A, 0, 0)
        grid_traj.addWidget(self.btn_move_A, 0, 1)
        grid_traj.addWidget(btn_save_B, 1, 0)
        grid_traj.addWidget(self.btn_move_B, 1, 1)
        grid_traj.addWidget(self.btn_reciprocate, 2, 0, 1, 2)
        left_layout.addLayout(grid_traj)
        
        # --- Obstacle Panel ---
        obs_group = QGroupBox("Obstacle Avoidance (Cylinder)")
        obs_layout = QGridLayout()
        
        obs_layout.addWidget(QLabel("Diameter:"), 0, 0)
        self.obs_d = QLineEdit("200")
        obs_layout.addWidget(self.obs_d, 0, 1)
        
        obs_layout.addWidget(QLabel("Height:"), 0, 2)
        self.obs_h = QLineEdit("400")
        obs_layout.addWidget(self.obs_h, 0, 3)
        
        obs_layout.addWidget(QLabel("X Center:"), 1, 0)
        self.obs_x = QLineEdit("400")
        obs_layout.addWidget(self.obs_x, 1, 1)
        
        obs_layout.addWidget(QLabel("Y Center:"), 1, 2)
        self.obs_y = QLineEdit("100")
        obs_layout.addWidget(self.obs_y, 1, 3)
        
        obs_layout.addWidget(QLabel("Z Base:"), 2, 0)
        self.obs_z = QLineEdit("0")
        obs_layout.addWidget(self.obs_z, 2, 1)
        
        obs_layout.addWidget(QLabel("Clearance Margin:"), 2, 2)
        self.obs_margin = QLineEdit("50")
        obs_layout.addWidget(self.obs_margin, 2, 3)
        
        self.btn_obstacle = QPushButton("Obstacle OFF")
        self.btn_obstacle.setCheckable(True)
        self.btn_obstacle.clicked.connect(self.toggle_obstacle)
        obs_layout.addWidget(self.btn_obstacle, 3, 0, 1, 4)
        
        # Bind inputs to dynamic recalculation
        for field in [self.obs_d, self.obs_h, self.obs_x, self.obs_y, self.obs_z, self.obs_margin]:
            field.editingFinished.connect(self.obstacle_params_changed)
            
        obs_group.setLayout(obs_layout)
        left_layout.addWidget(obs_group)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        left_layout.addWidget(self.status_label)
        
        scroll_area.setWidget(left_panel)
        
        # --- RIGHT PANEL (3D View) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        right_layout.addWidget(self.canvas)
        
        splitter.addWidget(scroll_area)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 1000])

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.anim_step)
        self.anim_index = 0

    def create_slider_row(self, label_text, min_val, max_val, slider_callback, jog_callback):
        layout = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(30)
        
        btn_left = QPushButton("<")
        btn_left.setFixedWidth(25)
        btn_left.clicked.connect(lambda: jog_callback(label_text, -1))
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val * 10, max_val * 10)
        slider.valueChanged.connect(lambda: slider_callback(label_text, slider.value() / 10.0))
        
        btn_right = QPushButton(">")
        btn_right.setFixedWidth(25)
        btn_right.clicked.connect(lambda: jog_callback(label_text, 1))
        
        input_field = QLineEdit("0.0")
        input_field.setFixedWidth(60)
        input_field.returnPressed.connect(lambda: slider_callback(label_text, float(input_field.text()), True))
        
        layout.addWidget(lbl)
        layout.addWidget(btn_left)
        layout.addWidget(slider)
        layout.addWidget(btn_right)
        layout.addWidget(input_field)
        return {'layout': layout, 'slider': slider, 'input': input_field, 'label': label_text}

    def get_jog_step(self):
        try: return float(self.jog_step_input.text())
        except ValueError: return 5.0

    def jog_fk(self, label, direction):
        step = self.get_jog_step()
        idx = int(label[1]) - 1
        current_val = float(self.fk_controls[idx]['input'].text())
        self.fk_controls[idx]['slider'].setValue(int((current_val + (direction * step)) * 10))

    def jog_ik(self, label, direction):
        step = self.get_jog_step()
        idx = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"].index(label)
        current_val = float(self.ik_controls[idx]['input'].text())
        self.ik_controls[idx]['slider'].setValue(int((current_val + (direction * step)) * 10))

    def apply_dh_params(self):
        try:
            for i in range(6):
                for j in range(4):
                    self.robot.dh_params[i, j] = float(self.dh_table.item(i, j).text())
            self.status_label.setText("Status: DH Parameters Applied.")
            self.update_plot()
        except ValueError: pass

    def fk_changed(self, label, val, from_input=False):
        if self.is_reciprocating: return
        idx = int(label[1]) - 1
        self.fk_controls[idx]['slider'].blockSignals(True)
        self.fk_controls[idx]['slider'].setValue(int(val * 10))
        self.fk_controls[idx]['slider'].blockSignals(False)
        self.fk_controls[idx]['input'].setText(f"{val:.1f}")
        self.robot.q[idx] = np.radians(val)
        self.update_ik_from_robot()
        self.update_plot()

    def ik_changed(self, label, val, from_input=False):
        if self.is_reciprocating: return
        idx = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"].index(label)
        self.ik_controls[idx]['slider'].blockSignals(True)
        self.ik_controls[idx]['slider'].setValue(int(val * 10))
        self.ik_controls[idx]['slider'].blockSignals(False)
        self.ik_controls[idx]['input'].setText(f"{val:.1f}")
        
        target_pos = np.array([float(self.ik_controls[i]['input'].text()) for i in range(3)])
        rpy = [np.radians(float(self.ik_controls[i]['input'].text())) for i in range(3, 6)]
        target_rot = euler_to_matrix(*rpy)
        
        new_q, success = self.robot.inverse_kinematics_dls(target_pos, target_rot, self.robot.q)
        if success:
            self.robot.q = new_q
            self.status_label.setText("Status: IK Solved.")
            self.update_fk_from_robot(update_ik=False)
            self.update_plot()
        else:
            self.status_label.setText("Status: Error - Singular or unreachable.")

    def update_fk_from_robot(self, update_ik=True):
        for i in range(6):
            deg_val = np.degrees(self.robot.q[i])
            self.fk_controls[i]['slider'].blockSignals(True)
            self.fk_controls[i]['slider'].setValue(int(deg_val * 10))
            self.fk_controls[i]['slider'].blockSignals(False)
            self.fk_controls[i]['input'].setText(f"{deg_val:.1f}")
        if update_ik: self.update_ik_from_robot()

    def update_ik_from_robot(self):
        T, _, _ = self.robot.forward_kinematics()
        pos = T[:3, 3]
        rpy_deg = np.degrees(matrix_to_euler(T[:3, :3]))
        vals = [pos[0], pos[1], pos[2], rpy_deg[0], rpy_deg[1], rpy_deg[2]]
        for i in range(6):
            self.ik_controls[i]['slider'].blockSignals(True)
            self.ik_controls[i]['slider'].setValue(int(vals[i] * 10))
            self.ik_controls[i]['slider'].blockSignals(False)
            self.ik_controls[i]['input'].setText(f"{vals[i]:.1f}")

    def get_current_pose(self):
        T, _, _ = self.robot.forward_kinematics()
        return {'pos': T[:3, 3], 'rot': T[:3, :3], 'q': np.copy(self.robot.q)}

    def save_point(self, point_name):
        if point_name == 'A':
            self.point_A = self.get_current_pose()
        else:
            self.point_B = self.get_current_pose()
        self.generate_trajectory()
        self.update_plot()

    def move_to_point(self, point_name):
        if self.is_reciprocating: return
        pt = self.point_A if point_name == 'A' else self.point_B
        if pt is not None:
            self.robot.q = np.copy(pt['q'])
            self.update_fk_from_robot()
            self.update_plot()

    def toggle_obstacle(self):
        self.obstacle_active = self.btn_obstacle.isChecked()
        self.btn_obstacle.setText("Obstacle ON" if self.obstacle_active else "Obstacle OFF")
        self.generate_trajectory()
        self.update_plot()
        
    def obstacle_params_changed(self):
        if self.obstacle_active:
            self.generate_trajectory()
            self.update_plot()

    def get_obstacle_params(self):
        try:
            d = float(self.obs_d.text())
            h = float(self.obs_h.text())
            x = float(self.obs_x.text())
            y = float(self.obs_y.text())
            z = float(self.obs_z.text())
            margin = float(self.obs_margin.text())
            # Ensure margin is considered in all dimensions explicitly
            return x, y, z, (d/2) + margin, h + margin
        except ValueError:
            return 0, 0, 0, 0, 0

    def check_cylinder_intersection(self, A, B, cx, cy, cz, R, H):
        ax, ay, az = A
        bx, by, bz = B
        
        dx, dy = bx - ax, by - ay
        fx, fy = ax - cx, ay - cy
        
        a = dx*dx + dy*dy
        b = 2 * (fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - R*R
        
        if a == 0: 
            if fx*fx + fy*fy <= R*R:
                min_z, max_z = min(az, bz), max(az, bz)
                if min_z <= cz + H and max_z >= cz:
                    return True
            return False
        
        discriminant = b*b - 4*a*c
        if discriminant >= 0:
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)
            
            t_candidates = []
            if 0 <= t1 <= 1: t_candidates.append(t1)
            if 0 <= t2 <= 1: t_candidates.append(t2)
            
            for t in t_candidates:
                z = az + t * (bz - az)
                if cz <= z <= cz + H:
                    return True
                    
            if t1 < 0 and t2 > 1:
                if (cz <= az <= cz + H) or (cz <= bz <= cz + H):
                    return True
                if (az < cz and bz > cz + H) or (bz < cz and az > cz + H):
                    return True
        return False

    def get_best_via_point(self, A, B, cx, cy, cz, R, H, rot_target, current_q):
        V = np.array([B[0]-A[0], B[1]-A[1], 0])
        norm_v = np.linalg.norm(V)
        if norm_v < 1e-3:
            N = np.array([1, 0, 0])
        else:
            V = V / norm_v
            N = np.array([-V[1], V[0], 0])
        
        mid_z = (A[2] + B[2]) / 2.0
        
        def find_safe_via(direction, start_dist, is_top=False):
            dist = start_dist
            for _ in range(30): # Iteratively push via point outwards until safety is strictly met
                if is_top:
                    vp = np.array([cx, cy, cz + dist])
                else:
                    vp = np.array([cx, cy, max(cz, mid_z)]) + direction * dist
                    
                if not self.check_cylinder_intersection(A, vp, cx, cy, cz, R, H) and \
                   not self.check_cylinder_intersection(vp, B, cx, cy, cz, R, H):
                    return vp
                dist += 20 
            return None

        # Start search strictly outside the expanded radius/height bounds
        vp_left = find_safe_via(N, R + 10)
        vp_right = find_safe_via(-N, R + 10)
        vp_top = find_safe_via(None, H + 10, is_top=True)
        
        candidates = [vp for vp in [vp_left, vp_right, vp_top] if vp is not None]
        
        best_vp = None
        best_manipulability = -1.0
        
        for vp in candidates:
            q_res, success = self.robot.inverse_kinematics_dls(vp, rot_target, current_q, max_iter=50)
            if success:
                manip = self.robot.get_manipulability(q_res)
                if manip > best_manipulability:
                    best_manipulability = manip
                    best_vp = vp
                    
        return best_vp

    def generate_trajectory(self):
        if self.point_A is None or self.point_B is None: return
        self.status_label.setText("Status: Planning Trajectory...")
        QApplication.processEvents()

        pos_A, rot_A = self.point_A['pos'], self.point_A['rot']
        pos_B, rot_B = self.point_B['pos'], self.point_B['rot']
        
        waypoints = [pos_A, pos_B]
        
        if self.obstacle_active:
            cx, cy, cz, R_expanded, H_expanded = self.get_obstacle_params()
            if self.check_cylinder_intersection(pos_A, pos_B, cx, cy, cz, R_expanded, H_expanded):
                mid_rot = euler_to_matrix(*((np.array(matrix_to_euler(rot_A)) + np.array(matrix_to_euler(rot_B))) / 2.0))
                via = self.get_best_via_point(pos_A, pos_B, cx, cy, cz, R_expanded, H_expanded, mid_rot, self.point_A['q'])
                if via is not None:
                    waypoints = [pos_A, via, pos_B]
                else:
                    self.status_label.setText("Error: Cannot find safe path around obstacle.")
                    self.btn_reciprocate.setEnabled(False)
                    self.drawn_trajectory = None
                    return

        num_steps_per_segment = 20
        self.trajectory_points = []
        cartesian_path = []
        current_q = self.point_A['q']
        
        valid_trajectory = True
        
        for w_idx in range(len(waypoints) - 1):
            start_pos = waypoints[w_idx]
            end_pos = waypoints[w_idx + 1]
            
            r_a, p_a, y_a = matrix_to_euler(rot_A)
            r_b, p_b, y_b = matrix_to_euler(rot_B)
            
            for i in range(num_steps_per_segment + 1):
                if w_idx > 0 and i == 0: continue
                
                t_seg = i / num_steps_per_segment
                interp_pos = start_pos * (1 - t_seg) + end_pos * t_seg
                
                global_t = (w_idx * num_steps_per_segment + i) / ((len(waypoints) - 1) * num_steps_per_segment)
                
                r = r_a * (1 - global_t) + r_b * global_t
                p = p_a * (1 - global_t) + p_b * global_t
                y = y_a * (1 - global_t) + y_b * global_t
                interp_rot = euler_to_matrix(r, p, y)
                
                q_res, success = self.robot.inverse_kinematics_dls(interp_pos, interp_rot, current_q)
                if success:
                    current_q = np.copy(q_res)
                    T, _, _ = self.robot.forward_kinematics(current_q)
                    cartesian_path.append(T[:3, 3])
                    self.trajectory_points.append(current_q)
                else:
                    valid_trajectory = False
                    break
            if not valid_trajectory: break
            
        if valid_trajectory:
            self.status_label.setText("Status: Trajectory planned successfully (Avoidance & Singularity checked).")
            self.btn_reciprocate.setEnabled(True)
            self.drawn_trajectory = np.array(cartesian_path)
        else:
            self.status_label.setText("Error: Path involves singularity or unreachability.")
            self.trajectory_points = None
            self.drawn_trajectory = None
            self.btn_reciprocate.setEnabled(False)

    def toggle_reciprocating(self):
        if self.trajectory_points is None: return
        if self.is_reciprocating:
            self.is_reciprocating = False
            self.anim_timer.stop()
            self.btn_reciprocate.setText("Reciprocating Motion")
            for ctrl in self.fk_controls + self.ik_controls:
                ctrl['slider'].setEnabled(True)
                ctrl['input'].setEnabled(True)
        else:
            self.is_reciprocating = True
            self.btn_reciprocate.setText("Stop Motion")
            for ctrl in self.fk_controls + self.ik_controls:
                ctrl['slider'].setEnabled(False)
                ctrl['input'].setEnabled(False)
            self.robot.q = np.copy(self.point_A['q'])
            self.anim_index = 0
            self.current_target = 'B'
            self.anim_timer.start(50)

    def anim_step(self):
        if self.current_target == 'B':
            self.anim_index += 1
            if self.anim_index >= len(self.trajectory_points) - 1:
                self.current_target = 'A'
        else:
            self.anim_index -= 1
            if self.anim_index <= 0:
                self.current_target = 'B'
                
        self.robot.q = np.copy(self.trajectory_points[self.anim_index])
        self.update_fk_from_robot()
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        T, positions, _ = self.robot.forward_kinematics()
        
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        self.ax.plot(x, y, z, '-o', color='b', linewidth=4, markersize=8, label='Robot Arm')
        self.ax.scatter(0, 0, 0, color='k', s=100, label='Base')
        
        if self.point_A is not None:
            pa = self.point_A['pos']
            self.ax.scatter(pa[0], pa[1], pa[2], color='g', s=100, marker='^', label='Point A')
            self.ax.text(pa[0], pa[1], pa[2]+50, 'A', color='g')
            
        if self.point_B is not None:
            pb = self.point_B['pos']
            self.ax.scatter(pb[0], pb[1], pb[2], color='r', s=100, marker='^', label='Point B')
            self.ax.text(pb[0], pb[1], pb[2]+50, 'B', color='r')
            
        if hasattr(self, 'drawn_trajectory') and self.drawn_trajectory is not None:
            self.ax.plot(self.drawn_trajectory[:, 0], self.drawn_trajectory[:, 1], self.drawn_trajectory[:, 2], 
                         linestyle='-.', color='magenta', linewidth=2, label='Trajectory')

        if self.obstacle_active:
            cx, cy, cz, r_exp, h_exp = self.get_obstacle_params()
            try:
                base_r = float(self.obs_d.text()) / 2.0
                base_h = float(self.obs_h.text())
            except ValueError:
                base_r = r_exp
                base_h = h_exp
            
            theta = np.linspace(0, 2*np.pi, 20)
            
            # Plot expanded boundary (margin)
            z_vals_exp = np.linspace(cz, cz + h_exp, 10)
            theta_grid, z_grid_exp = np.meshgrid(theta, z_vals_exp)
            x_exp = r_exp * np.cos(theta_grid) + cx
            y_exp = r_exp * np.sin(theta_grid) + cy
            self.ax.plot_surface(x_exp, y_exp, z_grid_exp, alpha=0.1, color='orange')
            
            # Plot physical obstacle
            z_vals_phys = np.linspace(cz, cz + base_h, 10)
            theta_grid, z_grid_phys = np.meshgrid(theta, z_vals_phys)
            x_phys = base_r * np.cos(theta_grid) + cx
            y_phys = base_r * np.sin(theta_grid) + cy
            self.ax.plot_surface(x_phys, y_phys, z_grid_phys, alpha=0.5, color='red')

        self.ax.set_xlim([-1000, 1000])
        self.ax.set_ylim([-1000, 1000])
        self.ax.set_zlim([0, 1200])
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title("3D Robot Visualization")
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())