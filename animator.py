'''Face animator'''
import os
import cv2
import copy
import socket
import threading
import trimesh
import pyrender
import numpy as np

from pyrender_mod import from_trimesh

class FaceAnimator(object):
    def __init__(self, ip, port, base_model_path, exp_bases_path, texture_path, eye_index_path=None):
        """Face animator and renderer, driven by coefficients from controller through TCP

        Args:
            ip (str): controller (server) IP
            port (int): controller (server) port
            base_model_path (str): path to the mean shape model (.obj)
            exp_bases_path (str): path to the expression bases (.npy)
            texture_path (str): path to the texture image (.jpg, .png, ...)
            eye_index_path (str, optional): path to the eye index file (.txt). Defaults to None.
        """
        self._init_base_model(base_model_path, texture_path)
        self._init_exp_bases(exp_bases_path, eye_index_path)
        self._init_pyrender()
        self._init_socket(ip, port)
        self._init_client_thread()
        
    def _init_base_model(self, base_model_path, texture_path):
        self.base_model_path = base_model_path
        self.texture_path = texture_path
        if os.path.isfile(self.base_model_path):
            # init model
            self.base_model = trimesh.load(self.base_model_path)
            self.normal = self.base_model.vertex_normals # use static normal
            self.mean_shape = self.base_model.vertices
            # init texture
            self.tex_img = cv2.cvtColor(cv2.imread(texture_path), cv2.COLOR_BGR2RGB)
        else:
            print(f"[Face Animation Client] Got wrong base model path: {base_model_path}")
            exit()
            
    def _init_exp_bases(self, exp_bases_path, eye_index_path):
        self.exp_bases = np.load(exp_bases_path)
        self.exp_num, self.vertices_num, _ = self.exp_bases.shape
        self.exp_bases = np.reshape(self.exp_bases, (self.exp_num, -1))
        if os.path.isfile(eye_index_path):
            self.eye_index = np.loadtxt(eye_index_path, dtype=np.int32)
        else:
            self.eye_index = None
            
    def _init_pyrender(self):
        # init shader texture
        tex = pyrender.Texture(source=self.tex_img, source_channels='RGB')
        self.mat = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=tex, wireframe=True)
        # init scene and view
        self.scene = pyrender.Scene()
        self.view = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True) # run rendering in a sub-thread
        # init light
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        sl = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0, innerConeAngle=0.05, outerConeAngle=0.5)
        self.scene.add(dl)
        self.scene.add(sl)
        # init model
        self.mesh = copy.deepcopy(self.base_model)
        mesh_tmp = pyrender.Mesh.from_trimesh(self.base_model, material=self.mat)
        self.node_buf = pyrender.Node(mesh=mesh_tmp, matrix=np.eye(4))
        self.scene.add_node(self.node_buf)
    
    def _init_socket(self, ip, port):
        self.ip = ip # ip and port should be the same as the server
        self.port = port
        self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.tcp_client.connect((self.ip, self.port))
        except Exception:
            print(f"[Face Animation Client] Can not connect to Server {ip}:{port}")
            self.view.close()
            exit()
        print(f"[Face Animation Client] Client connected to Server {ip}:{port}, ready to receive massages...")
        
    def _init_client_thread(self):
        self.client_thread = threading.Thread(target=self._recv_message)
        self.client_thread.setDaemon(True) # killed when main thread stops
        self.client_thread.start()
        
    def _recv_message(self):
        try:
            while True:
                data, addr = self.tcp_client.recvfrom(1024)
                self.raw_data = data.decode()
                print(f'\r[Face Animation Client] From Server: {data.decode()}', end='')
        except Exception as e:
            print(f"\n[Face Animation Client] Lose Connection with Server {self.ip}:{self.port}, error message: {e}")
            
    # update cycles
    
    def _update_vertices(self, eye_delta=0.01):
        try:
            # update coeff
            self.coeff_raw = np.array([float(x) for x in self.raw_data.split(',')])
            coeff = np.zeros(self.exp_num)
            coeff[0] = self.coeff_raw[25]/80
            coeff[1] = self.coeff_raw[13]/80
            coeff[2] = self.coeff_raw[12]/80
            # coeff[0] = self.coeff_raw[0]
            # coeff[1] = self.coeff_raw[1]
            # coeff[2] = self.coeff_raw[2]
            # update vertices (shape)
            v = (coeff @ self.exp_bases).reshape(self.vertices_num, 3) + self.mean_shape
            if self.eye_index is not None:
                for idx in self.eye_index:
                    v[idx] = self.mean_shape[idx]
                    v[idx][2] -= eye_delta
                return v
        except Exception:
            return self.mean_shape
    
    def _update_pyrender(self, vertices):
        self.mesh.vertices = vertices
        mesh_tmp = from_trimesh(self.mesh, self.normal, material=self.mat)
        self.view.render_lock.acquire()
        self.scene.remove_node(self.node_buf)
        self.node_buf = pyrender.Node(mesh=mesh_tmp, matrix=np.eye(4))
        self.scene.add_node(self.node_buf)
        self.view.render_lock.release()
        
    # user apis
            
    def start(self):
        while self.view.is_active:
            vertices = self._update_vertices()
            self._update_pyrender(vertices)
        self.tcp_client.close()
        print("[Face Animation Client] Client closed")
            
if __name__ == "__main__":
    fa = FaceAnimator("127.0.0.1", 985, "./sample/sample4/model_fine.obj", "./mlib/exp_bases.npy", "./sample/sample4/input.png", "./mlib/eye_idx.txt")
    fa.start()