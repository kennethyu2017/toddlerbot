# by:  https://gist.github.com/JeanElsner/755d0feb49864ecadab4ef00fd49a22b
#!/usr/bin/env python3
from collections import deque
import time
import mujoco
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow,
                               QVBoxLayout, QCheckBox, QGroupBox, QHBoxLayout)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtGui import QOpenGLFunctions
from PySide6.QtGui import QSurfaceFormat

from OpenGL import GL

XML = """<mujoco>
  <visual>
    <rgba haze=".3 .3 .3 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.6 0.6" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .25 .25" rgb2=".3 .3 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
  </asset>
  <worldbody>
    <geom name="floor" pos="0 0 0" size="0 0 1" type="plane" material="matplane"/>
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
    <body pos="0 0 .5" euler="45 45 0">
      <geom type="box" size=".1 .1 .1" rgba="1 1 1 .8" />
    </body>
  </worldbody>
</mujoco>"""

format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(format.DoubleBuffer)
format.setVersion(2,0)
# Deprecated
# format.setColorSpace(format.sRGBColorSpace)
format.setRenderableType(format.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)


class Viewport(QOpenGLWindow):

    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn) -> None:
        super().__init__()

        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt
        self.scn = scn

        self.width = 0
        self.height = 0
        self.__last_pos = None

        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1/60*1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def mousePressEvent(self, event):
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, action, dx / self.height, dy / self.height, self.scn, self.cam)
        self.__last_pos = pos

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005 * event.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h

    def paintGL(self) -> None:
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, self.scn, self.con)

        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):

    def __init__(self, model, data, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.running = True

    def run(self) -> None:
        while self.running:
            mujoco.mj_step(self.model, self.data)

    def stop(self):
        self.running = False
        self.wait()


class Window(QMainWindow):

    def __init__(self, XML) -> None:
        super().__init__()

        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QVBoxLayout()
        layout.addWidget(self.create_top())
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(640, 480)

        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps: float):
        self.statusBar().showMessage(f"Average runtime: {fps:.0e}s\t\
                                        Simulation time: {self.data.time:.0f}s")

    def create_top(self):
        layout = QHBoxLayout()
        collision_checkbox = QCheckBox("Reflection")
        collision_checkbox.stateChanged.connect(self.toggle_reflection)
        layout.addWidget(collision_checkbox)
        stereo_checkbox = QCheckBox("Shadow")
        stereo_checkbox.stateChanged.connect(self.toggle_shadow)
        layout.addWidget(stereo_checkbox)
        layout.addStretch()
        w = QGroupBox("Rendering")
        w.setLayout(layout)
        w.setFixedHeight(60)
        return w

    def toggle_shadow(self, state):
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(state)

    def toggle_reflection(self, state):
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = bool(state)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        for i in range(3):
            cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
        cam.distance = self.model.stat.extent
        cam.elevation = -45
        return cam


app = QApplication()
w = Window(XML)
w.show()
app.exec()
w.th.stop()
