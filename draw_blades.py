# -*- coding: utf-8 -*-

import numpy as np

from vispy import app,gloo

import h5py, os


# shaders.
# VS_base = """
# attribute vec2 a_position;
# void main() {
#     gl_Position = vec4(a_position, 0., 1.);
# }
# """

# FS_base = """
# uniform vec2 u_seeds[32];
# uniform vec3 u_colors[32];
# uniform vec2 u_screen;
# void main() {
#     float dist = distance(u_screen * u_seeds[0], gl_FragCoord.xy);
#     vec3 color = u_colors[0];
#     for (int i = 1; i < 32; i++) {
#         float current = distance(u_screen * u_seeds[i], gl_FragCoord.xy);
#         if (current < dist) {
#             color = u_colors[i];
#             dist = current;
#         }
#     }
#     gl_FragColor = vec4(color, 1.0);
# }
# """
VS_base = """
attribute vec2 a_position;
uniform float u_ps;
void main() {
    gl_Position = vec4(2. * a_position - 1., 0., 1.);
    gl_PointSize = 1. * u_ps;
}
"""

FS_base = """
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(1., 1., 1., 1.);
}
"""



class Canvas(app.Canvas):
    def __init__(self,coordinates,indices):
        app.Canvas.__init__(self, size=(600, 600), title='Draw Blade',
                            keys='interactive')

        self.ps = self.pixel_scale

        # self.seeds = np.random.uniform(0, 1.0 * self.ps,
        #                                size=(32, 2)).astype(np.float32)
        # self.colors_red = np.array([1.0,0.0,0.0]).astype(np.float32)
        # self.colors_white = np.array([0.0,0.0,0.0]).astype(np.float32)
        self.indices_buffer = gloo.IndexBuffer(indices)

        # Set Voronoi program.
        # self.program_v = gloo.Program(VS_base, FS_base)
        # self.program_v['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        # # HACK: work-around a bug related to uniform arrays until
        # # issue #345 is solved.
        # for i in range(32):
        #     self.program_v['u_seeds[%d]' % i] = self.seeds[i, :]
        #     self.program_v['u_colors[%d]' % i] = self.colors[i, :]

        # Set seed points program.
        self.program_s = gloo.Program(VS_base, FS_base)
        self.program_s['a_position'] = coordinates
        self.program_s['u_ps'] = self.ps

        self.activate_zoom()

        self.show()


    def on_draw(self, event):
        gloo.clear()
        self.program_s.draw('points')
        # self.program_s.draw('lines',indices=self.indices_buffer)

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)
        # self.program_v['u_screen'] = self.physical_size


if __name__ == "__main__":
    file_name = "./test.cgns"
    print(file_name)
    f = h5py.File(file_name,'r')
    x = f['/Base/Zone/GridCoordinates/CoordinateX/ data'][:]
    y = f['/Base/Zone/GridCoordinates/CoordinateY/ data'][:]
    ratio = 1/(max(x)-min(x))
    x = (x-min(x)) * ratio * 0.9 + 0.05
    y = (y-min(y)) * ratio * 0.9 + 0.5
    print(min(y),max(y))
    coordinates = np.stack([x,y],axis=1).astype(np.float32)
    cells_unspecified = f['/Base/Zone/unspecified/ElementConnectivity/ data'][:]
    cells_inlet = f['/Base/Zone/periodic/ElementConnectivity/ data'][:]
    # cells_outlet = f['/Base/Zone/outlet/ElementConnectivity/ data'][:]
    # cells_periodic = f['/Base/Zone/periodic/ElementConnectivity/ data'][:]
    # cells_inlet = cells_inlet.reshape(-1,3)
    # cells_inlet = cells_inlet[:,1:].astype(np.uint32)
    # cells_inlet -= 1
    cells_unspecified = cells_unspecified.reshape(-1,5)[:,1:].astype(np.uint32)
    cells_unspecified -= 1
    print(np.min(cells_unspecified),np.max(cells_unspecified))
    l1 = cells_unspecified[:,0:2]
    l2 = cells_unspecified[:,1:3]
    l3 = cells_unspecified[:,2:4]
    l4 = np.stack([cells_unspecified[:,0],cells_unspecified[:,3]],axis=1)
    cells = np.concatenate([l1,l2,l3,l4])
    print(cells.shape)
    c = Canvas(coordinates[500:501,:],cells)
    c = Canvas(coordinates[3968:3969,:],cells)
    app.run()