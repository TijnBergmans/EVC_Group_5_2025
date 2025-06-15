#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Person Re-ID:

Subscribes to /obj_detect/person_detect (`final/DetectedPerson`).
Builds a template embedding from `~ref_dir` (images of the target).
Prints similarity for every snippet, flags detections ≥ `~sim_threshold`.
Optional crop saving (`~save_enabled`).
Publishes to **/target_position**
"""

from __future__ import division, print_function
import os, glob, time
import numpy as np
import cv2, rospy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage
from final.msg import DetectedPerson

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # default context

CUDA_CTX   = pycuda.autoinit.context
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

IMG_H, IMG_W = 256, 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# TensorRT helpers

def load_engine(path):
    if not os.path.isfile(path):
        rospy.logfatal('[ReID] TensorRT engine not found: %s', path)
        raise SystemExit(1)
    with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as rt:
        return rt.deserialize_cuda_engine(f.read())

def alloc_buffers(engine):
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for name in engine:
        size  = int(trt.volume(engine.get_binding_shape(name)))
        dtype = trt.nptype(engine.get_binding_dtype(name))
        host  = cuda.pagelocked_empty(size, dtype)
        dev   = cuda.mem_alloc(host.nbytes)
        bindings.append(int(dev))
        (inputs if engine.binding_is_input(name) else outputs).append({'host': host, 'device': dev})
    return inputs, outputs, bindings, stream

def infer(ctx, bindings, inputs, outputs, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    ctx.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()
    return [out['host'] for out in outputs]

def preprocess(bgr):
    img = cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1).copy()

class PersonReIDNode(object):
    def __init__(self):
        rospy.init_node('person_reid_node')

        # parameters
        self.engine_path   = rospy.get_param('engine')
        self.ref_dir       = rospy.get_param('ref_dir')
        self.save_dir      = rospy.get_param('save_dir', os.path.expanduser('~/reid_snippets'))
        self.save_enabled  = rospy.get_param('save_enabled', True)
        self.SIM_THRESHOLD = rospy.get_param('sim_threshold', 0.3)
        self.pub_rate      = rospy.get_param('pub_rate', 0.2)
        self.hold_time     = rospy.get_param('hold_time', 0.5)

        # create save dir if needed
        if self.save_enabled and not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # TensorRT setup
        engine = load_engine(self.engine_path)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = alloc_buffers(engine)

        # Build reference embedding
        self.ref_vec = self._build_template()
        rospy.loginfo('[ReID] Node ready (thr=%.2f, save=%s, pub=%.1fs)'.decode('utf-8'),
                      self.SIM_THRESHOLD, self.save_enabled, self.pub_rate)

        # Publisher & state
        self.pub   = rospy.Publisher('/target_position', Float32MultiArray, queue_size=1)
        self.last_bbox  = [0.0, 0.0, 0.0, 0.0]
        self.last_time  = 0.0

        # ROS interfaces
        rospy.Subscriber('/obj_detect/person_detect', DetectedPerson,
                         self.person_cb, queue_size=1, buff_size=2**22)
        rospy.Timer(rospy.Duration(self.pub_rate), self.timer_cb)

    def _embed(self, bgr):
        img = preprocess(bgr)
        np.copyto(self.inputs[0]['host'], img.ravel())
        feat = infer(self.context, self.bindings, self.inputs, self.outputs, self.stream)[0]
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    def _build_template(self):
        files = sum([glob.glob(os.path.join(self.ref_dir, e)) for e in ('*.jpg','*.png','*.jpeg','*.bmp')], [])
        if not files:
            rospy.logfatal('[ReID] No reference images in %s', self.ref_dir)
            raise SystemExit(1)
        feats = [self._embed(cv2.imread(p)) for p in files]
        vec   = np.mean(feats, axis=0)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def person_cb(self, msg):
        
        timer_start = time.time()

        CUDA_CTX.push()
        try:
            crop = cv2.imdecode(np.frombuffer(msg.image.data, np.uint8), cv2.IMREAD_COLOR)
            if crop is None:
                return

            # embed & compare
            sim = float(np.dot(self.ref_vec, self._embed(crop)))
            rospy.loginfo('[ReID] similarity=%.3f (thr=%.2f)', sim, self.SIM_THRESHOLD)

            if sim >= self.SIM_THRESHOLD:
                bb = msg.bbox
                self.last_bbox = [float(bb.x_offset), float(bb.y_offset), float(bb.width), float(bb.height)]
                self.last_time = time.time()

                rospy.loginfo('[ReID] Target DETECTED @ [x=%d y=%d w=%d h=%d] sim=%.3f',
                              bb.x_offset, bb.y_offset, bb.width, bb.height, sim)

                if self.save_enabled:
                    fname = os.path.join(self.save_dir, 'crop_%d.jpg' % int(self.last_time*1000))
                    cv2.imwrite(fname, crop)
        finally:
            CUDA_CTX.pop()

        proc_time = time.time() - timer_start
        rospy.loginfo("ID Time: %.4f sec", proc_time)

    def timer_cb(self, _):
        msg = Float32MultiArray()
        if time.time() - self.last_time <= self.hold_time:
            msg.data = self.last_bbox
        else:
            msg.data = [0.0, 0.0, 0.0, 0.0]
        self.pub.publish(msg)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        PersonReIDNode().spin()
    except rospy.ROSInterruptException:
        pass
